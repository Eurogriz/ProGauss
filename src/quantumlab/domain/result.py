"""Модель результата расчёта.

Принцип хранения (§24 ТЗ): в SQL живут **скаляры и метаданные** (энергия,
диполь, статусы, проверки качества), а большие массивы (градиенты, гессианы,
орбитали, кубические сетки) — в объектном хранилище, на которые ссылаются
:class:`ArtifactRef`. Никогда не кладём гигантские бинарные данные в таблицы.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from quantumlab.domain.fingerprint import Fingerprint
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import CalculationSpec


class ArtifactKind(StrEnum):
    """Тип научного артефакта."""

    GRADIENT = "gradient"
    HESSIAN = "hessian"
    ORBITALS = "orbitals"
    DENSITY = "density"
    SPIN_DENSITY = "spin_density"
    ESP = "esp"
    FREQUENCIES = "frequencies"
    NORMAL_MODES = "normal_modes"
    CHECKPOINT = "checkpoint"
    RAW_LOG = "raw_log"
    REPORT = "report"


class ArtifactRef(BaseModel):
    """Ссылка на артефакт в объектном хранилище."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ArtifactKind
    uri: str = Field(description="Схема: artifact://<bucket>/<key>")
    sha256: str = Field(description="Контрольная сумма содержимого")
    size_bytes: int = Field(ge=0)
    schema_version: str = Field(description="Версия схемы артефакта")


class OrbitalInfo(BaseModel):
    """Сведения об одной молекулярной орбитали."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0, description="0-основная нумерация")
    energy_hartree: float
    occupation: float = Field(ge=0.0, le=2.0)
    symmetry_label: str | None = None


class TimingRecord(BaseModel):
    """Замер времени этапа — вход для benchmark-системы (§27 ТЗ)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(description="Например 'integrals', 'scf', 'diagonalization'")
    wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float = Field(ge=0.0)


class EnvironmentInfo(BaseModel):
    """Программно-аппаратное окружение расчёта (§40 ТЗ)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    software_version: str
    engine_version: str
    engine_backend: str = Field(description="Например 'reference-cpu', 'cuda'")
    python_version: str
    os: str
    hostname: str
    cpu_model: str
    cores: int = Field(ge=1)
    memory_mb: int = Field(ge=0)
    gpu: str | None = None
    mpi_ranks: int = 1


class QualityVerdict(StrEnum):
    """Вердикт проверки качества (§28 ТЗ)."""

    PASS = "pass"
    WARNING = "warn"
    FAIL = "fail"
    NOT_CHECKED = "not_checked"


class QualityCheck(BaseModel):
    """Одна проверка качества с локализуемым названием."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name_key: str = Field(description="Ключ локализации, например 'quality.scf_converged'")
    verdict: QualityVerdict
    detail: str | None = None


class CalculationWarning(BaseModel):
    """Предупреждение расчёта: ключ перевода и подстановки, а не готовая строка.

    Движок не знает языка пользователя, поэтому возвращает ``key`` и ``params``,
    а текст собирает граница (CLI, REST) в нужной локали. Хранить в результате
    русскую строку означало бы, что английский интерфейс либо её не покажет,
    либо получит второй, способный разойтись перевод того же факта (§3 ТЗ).

    Параметры — строки: предупреждение не место для арифметики, форматирование
    числа выполняется там, где возник факт, и в результат уходит готовая
    подстановка.
    """

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Dotted-ключ каталога переводов (``warning.*``).")
    params: dict[str, str] = Field(
        default_factory=dict, description="Подстановки для плейсхолдеров шаблона."
    )


class CalculationResult(BaseModel):
    """Результат завершённого расчёта.

    Все энергии — в хартри, частоты — в см⁻¹, диполь — в дебаях (это
    единственное исключение: дебай привычнее пользователю, и в UI это
    указано явно).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    spec: CalculationSpec
    fingerprint: Fingerprint
    energy_hartree: float = Field(
        description=(
            "Полная энергия: электронная (SCF) плюс дисперсионная поправка, "
            "если она была запрошена и применена. Вклад поправки виден "
            "отдельно в ``dispersion_energy_hartree``."
        )
    )
    dispersion_energy_hartree: float | None = Field(
        default=None,
        description=(
            "Вклад дисперсионной поправки D3 в энергию (отрицательный), "
            "хартри. ``None`` — поправка не запрашивалась. Поле присутствует, "
            "чтобы энергия с D3 не выдавалась за энергию без неё: разница "
            "видна пользователю, а не теряется в полной сумме (§8, §54 ТЗ)."
        ),
    )
    scf_iterations: int = Field(default=0, ge=0)
    converged: bool = False
    homo_energy_hartree: float | None = Field(
        default=None,
        description=(
            "Энергия ВЗМО. Для UHF — канал α (β вынесен в отдельные поля: "
            "у открытой оболочки две независимые системы орбиталей)."
        ),
    )
    lumo_energy_hartree: float | None = Field(
        default=None, description="Энергия НСМО; для UHF — канал α."
    )
    gap_hartree: float | None = None
    beta_homo_energy_hartree: float | None = Field(
        default=None,
        description="Энергия ВЗМО канала β. Заполняется только для UHF.",
    )
    beta_lumo_energy_hartree: float | None = Field(
        default=None,
        description="Энергия НСМО канала β. Заполняется только для UHF.",
    )
    spin_squared: float | None = Field(
        default=None,
        description=(
            "Ожидание <S^2>. Заполняется только для UHF: там возможно спиновое "
            "загрязнение, и скрывать его нельзя. Для чистого состояния равно S(S+1)."
        ),
    )
    dipole_debye: float | None = None
    frequencies_cm1: tuple[float, ...] = ()
    zero_point_energy_hartree: float | None = None
    orbitals: tuple[OrbitalInfo, ...] = ()
    optimization_steps: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Число шагов оптимизации геометрии. Пусто для расчётов, которые структуру не меняют."
        ),
    )
    final_molecule: Molecule | None = Field(
        default=None,
        description=(
            "Итоговая геометрия, к которой относятся приведённые числа. "
            "Заполняется расчётами, меняющими структуру (оптимизация); для "
            "одноточечного расчёта остаётся пустым — дублировать исходную "
            "структуру смысла нет."
        ),
    )
    quality_checks: tuple[QualityCheck, ...] = ()
    timings: tuple[TimingRecord, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[CalculationWarning, ...] = ()
    environment: EnvironmentInfo
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def wall_seconds(self) -> float:
        """Суммарное wall time всех этапов."""
        return sum(record.wall_seconds for record in self.timings)

    def checks_by_name(self) -> dict[str, QualityCheck]:
        """Проверки качества, индексированные по ключу."""
        return {check.name_key: check for check in self.quality_checks}

    def artifact(self, kind: ArtifactKind) -> ArtifactRef | None:
        """Первая ссылка на артефакт заданного типа."""
        for ref in self.artifacts:
            if ref.kind is kind:
                return ref
        return None
