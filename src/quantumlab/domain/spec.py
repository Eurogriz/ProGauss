"""Спецификация расчёта — единый входной контракт для всех интерфейсов.

Один и тот же :class:`CalculationSpec` создают GUI (мастер), CLI, REST API и
Python SDK, и один и тот же объект попадает в журнал аудита, чекпоинт и
отпечаток воспроизводимости (§40 ТЗ). Поэтому здесь не бывает «полей только
для GUI»: всё, что влияет на результат, описано явно.

Два уровня настроек (§35, §36 ТЗ):

* **базовый** — ``task`` + ``profile`` (профиль точности), остальное подбирается
  автоматически и показывается пользователю как список обоснований;
* **экспертный** — явные ``method``, ``scf``, ``grid``, ``optimization``,
  ``resources``; они переопределяют автоподбор.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(StrEnum):
    """Тип квантовохимической задачи (§9 ТЗ)."""

    SINGLE_POINT = "single_point"
    OPTIMIZATION = "optimization"
    FREQUENCIES = "frequencies"
    TS_OPTIMIZATION = "ts_optimization"
    IRC = "irc"
    SCAN_1D = "scan_1d"
    SCAN_2D = "scan_2d"
    PROPERTIES = "properties"

    @property
    def i18n_key(self) -> str:
        """Ключ локализации названия задачи."""
        return f"task.{self.value}.title"


class TheoryFamily(StrEnum):
    """Семейство методов (§5 ТЗ)."""

    HF = "hf"
    DFT = "dft"
    MP2 = "mp2"
    SCS_MP2 = "scs_mp2"
    CCSD = "ccsd"
    CCSD_T = "ccsd_t"


class FunctionalClass(StrEnum):
    """Класс обменно-корреляционного функционала (§5 ТЗ)."""

    LDA = "lda"
    GGA = "gga"
    MGGA = "mgga"
    HYBRID = "hybrid"
    RANGE_SEPARATED_HYBRID = "range_separated_hybrid"
    DOUBLE_HYBRID = "double_hybrid"


class SpinTreatment(StrEnum):
    """Обработка спина: закрытая/открытая оболочка."""

    RHF = "rhf"
    UHF = "uhf"
    ROHF = "rohf"


class DispersionCorrection(StrEnum):
    """Дисперсионная поправка."""

    NONE = "none"
    D3_BJ = "d3bj"
    D3_ZERO = "d3zero"
    D4 = "d4"


class GridPreset(StrEnum):
    """Именованные квадратурные сетки DFT.

    Числа точек намеренно не зашиты в enum: конкретные схемы (Lebedev,
    гауссовы радиальные) выбирает движок, а спецификация фиксирует только
    уровень точности — иначе отпечаток расчёта ломался бы при улучшении сетки.
    """

    COARSE = "coarse"
    FINE = "fine"
    ULTRAFINE = "ultrafine"


class DeviceKind(StrEnum):
    """Вычислительное устройство (§7 ТЗ)."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


class PrecisionProfile(StrEnum):
    """Профили точности «Рекомендуемых настроек» (§8 ТЗ)."""

    SCREENING = "screening"
    STANDARD = "standard"
    HIGH_ACCURACY = "high_accuracy"
    RESEARCH = "research"

    @property
    def i18n_key(self) -> str:
        """Ключ локализации названия профиля."""
        return f"profile.{self.value}.name"


class MethodSpec(BaseModel):
    """Метод: семейство теории, функционал, базис, дисперсия, спин."""

    model_config = ConfigDict(extra="forbid")

    theory: TheoryFamily = TheoryFamily.DFT
    functional: str | None = Field(
        default=None, description="Например 'pbe0'; None — только для HF"
    )
    functional_class: FunctionalClass | None = None
    basis: str = Field(description="Название базисного набора из реестра")
    dispersion: DispersionCorrection = DispersionCorrection.NONE
    spin: SpinTreatment = SpinTreatment.RHF

    @model_validator(mode="after")
    def _check_functional(self) -> MethodSpec:
        if self.theory is TheoryFamily.DFT and not self.functional:
            msg = "Для DFT-расчёта необходимо указать обменно-корреляционный функционал"
            raise ValueError(msg)
        if self.theory is TheoryFamily.HF and self.functional:
            msg = "Метод Хартри–Фока не использует обменно-корреляционный функционал"
            raise ValueError(msg)
        return self


class ScfSpec(BaseModel):
    """Параметры SCF-движка (§10 ТЗ).

    ``fallback_strategies`` — упорядоченный список стратегий, которые движок
    применяет при отсутствии сходимости. Порядок важен: от дешёвых к дорогим.
    Каждая применённая стратегия попадает в журнал и в диагноз ошибки.
    """

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=128, ge=1)
    energy_threshold: float = Field(default=1e-8, gt=0.0, description="Порог по энергии, Eh")
    density_threshold: float = Field(
        default=1e-6, gt=0.0, description="Порог по плотности/ошибкам матриц"
    )
    diis_start: int = Field(default=1, ge=0, description="Итерация, с которой включается DIIS")
    damping: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Доля старой плотности (0 — выключено)"
    )
    level_shift: float = Field(default=0.0, ge=0.0, description="Сдвиг виртуальных орбиталей, Eh")
    stability_analysis: bool = Field(
        default=False, description="Проверка устойчивости волновой функции"
    )
    fractional_occupations: bool = Field(
        default=False, description="Дробные заселённости (smearing)"
    )
    fallback_strategies: tuple[str, ...] = ("ediis", "damping", "level_shift", "soscf")


class GridSpec(BaseModel):
    """Квадратурная сетка DFT (§8 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    preset: GridPreset = GridPreset.FINE
    prune: bool = Field(default=True, description="Прореживание угловой сетки вблизи ядра")


class CoordinateConstraint(BaseModel):
    """Ограничение на внутреннюю координату (§9 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    atoms: tuple[int, ...] = Field(
        min_length=2, max_length=4, description="2 — связь, 3 — угол, 4 — двугранный угол"
    )
    value: float | None = Field(
        default=None, description="Фиксированное значение; None — только сканирование"
    )
    frozen: bool = False


class OptimizationSpec(BaseModel):
    """Параметры геометрической оптимизации (§11 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    coordinates: str = Field(
        default="redundant_internal", pattern="^(cartesian|internal|redundant_internal)$"
    )
    max_steps: int = Field(default=100, ge=1)
    max_force: float = Field(default=4.5e-4, gt=0.0, description="Eh/Bohr")
    rms_force: float = Field(default=3.0e-4, gt=0.0, description="Eh/Bohr")
    max_displacement: float = Field(default=1.8e-3, gt=0.0, description="Bohr")
    rms_displacement: float = Field(default=1.2e-3, gt=0.0, description="Bohr")
    trust_radius: float = Field(default=0.3, gt=0.0, description="Bohr")
    hessian_update: str = Field(default="bfgs", pattern="^(bfgs|bofill|none)$")
    frozen_atoms: tuple[int, ...] = ()
    constraints: tuple[CoordinateConstraint, ...] = ()


class ScanSpec(BaseModel):
    """Параметры сканирования координаты (§9 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    coordinate: CoordinateConstraint
    start: float
    stop: float
    steps: int = Field(default=10, ge=1)


class ResourceSpec(BaseModel):
    """Ресурсы выполнения (§7, §12 ТЗ).

    ``None`` означает «решить автоматически»: планировщик подбирает значение по
    размеру задачи и загрузке системы и показывает выбор пользователю.
    """

    model_config = ConfigDict(extra="forbid")

    threads: int | None = Field(default=None, ge=1)
    memory_mb: int | None = Field(default=None, ge=1)
    device: DeviceKind = DeviceKind.AUTO
    nodes: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=0, ge=0)
    scheduler: str | None = Field(default=None, description="local | slurm | pbs | lsf")
    wall_time_minutes: int | None = Field(default=None, ge=1)


class CalculationSpec(BaseModel):
    """Полная спецификация расчёта."""

    model_config = ConfigDict(extra="forbid")

    task: Task = Task.SINGLE_POINT
    profile: PrecisionProfile | None = Field(
        default=PrecisionProfile.STANDARD,
        description="Профиль автоподбора; None — все параметры заданы вручную",
    )
    method: MethodSpec | None = Field(
        default=None, description="Явное задание метода (экспертный режим)"
    )
    scf: ScfSpec = Field(default_factory=ScfSpec)
    grid: GridSpec = Field(default_factory=GridSpec)
    optimization: OptimizationSpec = Field(default_factory=OptimizationSpec)
    scan: ScanSpec | None = None
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    seed: int | None = Field(default=None, description="Зерно ГСЧ, если метод стохастический")
    expert_raw_input: str | None = Field(
        default=None, description="Сырой input для внешнего движка (§36 ТЗ)"
    )

    @model_validator(mode="after")
    def _check_task_requirements(self) -> CalculationSpec:
        if self.task in (Task.SCAN_1D, Task.SCAN_2D) and self.scan is None:
            msg = "Для задачи сканирования необходимо задать параметры сканирования"
            raise ValueError(msg)
        if self.profile is None and self.method is None and self.expert_raw_input is None:
            msg = "Нужно указать профиль автоподбора, явный метод или сырой input"
            raise ValueError(msg)
        return self

    # -- сериализация и отпечаток ------------------------------------------- #
    def canonical_json(self) -> str:
        """Канонический JSON: сортированные ключи, без лишних пробелов.

        Именно этот текст хешируется в отпечаток расчёта, поэтому любые
        изменения схемы спецификации меняют отпечаток — это ожидаемое поведение.
        """
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def describe_decisions(self, decisions: list[tuple[str, object]]) -> dict[str, Any]:
        """Вспомогательный метод: превращает решения автоподбора в словарь для UI."""
        return {"decisions": [{"parameter": name, "value": value} for name, value in decisions]}
