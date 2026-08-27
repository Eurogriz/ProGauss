"""Автоподбор параметров расчёта — «Рекомендуемые настройки» (§8 ТЗ).

Пользователь выбирает уровень точности, система сама выбирает метод, базис,
пороги, сетку и ресурсы, и **обязана показать каждое своё решение** с
обоснованием. Поэтому функция возвращает не только спецификацию, но и список
:class:`Decision`, который UI рендерит в блоке «Почему выбраны эти параметры».

Эвристики намеренно просты и задокументированы: их задача — дать надёжную
отправную точку, а не заменить специалиста. Любое значение можно перекрыть в
экспертном режиме (§36 ТЗ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    DeviceKind,
    DispersionCorrection,
    FunctionalClass,
    GridPreset,
    GridSpec,
    MethodSpec,
    OptimizationSpec,
    PrecisionProfile,
    ResourceSpec,
    ScfSpec,
    Task,
    TheoryFamily,
)
from quantumlab.i18n import DEFAULT_LOCALE, t

#: Число базисных функций на атом — грубая оценка для выбора ресурсов.
#: SVP-подобные базисы ≈ 5, TZVP-подобные ≈ 10 функций на атом.
_BASIS_FUNCTIONS_PER_ATOM: Final[dict[str, int]] = {
    "def2-svp": 5,
    "def2-tzvp": 10,
    "def2-tzvpp": 11,
    "def2-qzvp": 20,
    "6-31g(d)": 5,
    "6-31g(d,p)": 6,
    "sto-3g": 2,
}

#: Порог «крупная молекула», после которого точность обменивается на время.
_LARGE_SYSTEM_ATOMS: Final = 80

#: Единственная реализованная система координат оптимизации. Реестр
#: возможностей сообщает о том же (``coordinates:cartesian`` = partial).
_COORDINATES: Final = "cartesian"


@dataclass(frozen=True, slots=True)
class Decision:
    """Одно автоматическое решение с обоснованием.

    Attributes:
        parameter: машиночитаемое имя параметра (``basis``, ``threads``, …).
        value: выбранное значение в виде строки для показа.
        reason_key: ключ локализации фразы «Выбран базис {value}».
        detail: дополнительная причина (например, размер системы).
    """

    parameter: str
    value: str
    reason_key: str
    detail: str | None = None

    def render(self, locale: str = DEFAULT_LOCALE) -> str:
        """Локализованная строка решения для показа пользователю."""
        line = t(self.reason_key, locale, value=self.value)
        return f"{line} — {self.detail}" if self.detail else line


@dataclass(frozen=True, slots=True)
class HardwareContext:
    """Ресурсы, доступные планировщику в момент подбора."""

    cores: int = 1
    memory_mb: int = 4096
    gpu_count: int = 0


@dataclass(frozen=True, slots=True)
class Resolution:
    """Результат автоподбора: готовая спецификация + объяснения."""

    spec: CalculationSpec
    decisions: tuple[Decision, ...]

    def explain(self, locale: str = DEFAULT_LOCALE) -> list[str]:
        """Строки для блока «Почему выбраны эти параметры»."""
        profile_name = (
            t(f"profile.{self.spec.profile.value}.name", locale) if self.spec.profile else ""
        )
        return [
            t("profile.because", locale, profile=profile_name),
            *(d.render(locale) for d in self.decisions),
        ]


def _basis_functions(basis: str, n_atoms: int) -> int:
    return _BASIS_FUNCTIONS_PER_ATOM.get(basis, 7) * n_atoms


def _estimate_threads(n_basis_functions: int, cores: int) -> int:
    """Оценка числа потоков.

    Модель: плотные BLAS/интегральные ядра выходят на плато примерно при одном
    потоке на 60 базисных функций (дальше упираемся в пропускную способность
    памяти). Оценка намеренно консервативна: недоиспользовать ядро дешевле,
    чем убить задание по памяти.
    """
    target = math.ceil(n_basis_functions / 60)
    return max(1, min(cores, target))


def _estimate_memory_mb(n_basis_functions: int) -> int:
    """Оценка потребности в памяти, МБ.

    Учитываем плотные матрицы SCF (F, P, S, X, D, ошибки DIIS ≈ 6 штук) по
    ``N_bf²`` элементов ``float64``; четырёхиндексные интегралы (N⁴) в оценке
    не участвуют, потому что по умолчанию используется direct-SCF. Для
    N_bf = 1000 это ≈ 48 ГБ — честная иллюстрация того, почему прямой SCF
    обязателен для крупных систем.
    """
    dense_bytes = 6 * n_basis_functions * n_basis_functions * 8
    return max(1024, math.ceil(dense_bytes / 1_000_000))


def resolve_profile(
    profile: PrecisionProfile,
    *,
    task: Task,
    molecule: Molecule,
    hardware: HardwareContext | None = None,
) -> Resolution:
    """Подбирает полную спецификацию расчёта по профилю точности.

    Args:
        profile: уровень точности, выбранный пользователем.
        task: тип задачи.
        molecule: структура (нужна для оценки размера).
        hardware: доступные ресурсы; если не заданы — берутся консервативные.

    Returns:
        :class:`Resolution` со спецификацией и списком обоснований.
    """
    hardware = hardware or HardwareContext()
    decisions: list[Decision] = []

    functional, functional_class, basis, dispersion = _base_choice(profile)
    decisions.append(Decision("functional", functional, "profile.decision.functional"))

    is_large = molecule.n_atoms > _LARGE_SYSTEM_ATOMS
    if is_large and basis != "def2-svp":
        basis = "def2-svp"
        decisions.append(
            Decision(
                "basis",
                basis,
                "profile.decision.basis",
                detail=f"{molecule.n_atoms} атомов — базис уменьшен ради времени счёта",
            )
        )
    else:
        decisions.append(Decision("basis", basis, "profile.decision.basis"))

    decisions.append(Decision("dispersion", dispersion.value, "profile.decision.dispersion"))

    grid_preset, scf = _numerics(profile, task, is_large)
    decisions.append(Decision("grid", grid_preset.value, "profile.decision.grid"))
    decisions.append(
        Decision("scf_threshold", f"{scf.energy_threshold:.0e}", "profile.decision.scf_threshold")
    )
    decisions.append(
        Decision(
            "integral_threshold",
            f"{_integral_threshold(profile, is_large):.0e}",
            "profile.decision.integral_threshold",
        )
    )

    n_basis_functions = _basis_functions(basis, molecule.n_atoms)
    threads = _estimate_threads(n_basis_functions, hardware.cores)
    memory_request = _estimate_memory_mb(n_basis_functions)
    memory_mb = min(memory_request, hardware.memory_mb)
    decisions.append(Decision("threads", str(threads), "profile.decision.threads"))
    decisions.append(
        Decision(
            "memory",
            f"{memory_mb} МБ",
            "profile.decision.memory",
            detail=f"оценка по {n_basis_functions} базисным функциям",
        )
    )
    device = DeviceKind.AUTO
    decisions.append(
        Decision(
            "device",
            device.value,
            "profile.decision.device",
            detail=f"доступно GPU: {hardware.gpu_count}",
        )
    )

    if task in (Task.OPTIMIZATION, Task.TS_OPTIMIZATION):
        # Координаты оптимизации выбираются явно и с объяснением: дефолт
        # спецификации — избыточные внутренние координаты, которых в ядре пока
        # нет. Молча подставить декартовы нельзя (§8 ТЗ): у них другая скорость
        # сходимости, и пользователь должен это видеть.
        decisions.append(
            Decision(
                "coordinates",
                _COORDINATES,
                "profile.decision.coordinates",
                detail=(
                    "избыточные внутренние координаты ещё не реализованы, "
                    "поэтому расчёт идёт в декартовых — сходимость может "
                    "потребовать больше шагов"
                ),
            )
        )

    spec = CalculationSpec(
        task=task,
        profile=profile,
        method=MethodSpec(
            theory=TheoryFamily.DFT,
            functional=functional,
            functional_class=functional_class,
            basis=basis,
            dispersion=dispersion,
        ),
        scf=scf,
        grid=GridSpec(preset=grid_preset),
        optimization=_optimization(task),
        resources=ResourceSpec(threads=threads, memory_mb=memory_mb, device=device),
    )
    return Resolution(spec=spec, decisions=tuple(decisions))


def _base_choice(
    profile: PrecisionProfile,
) -> tuple[str, FunctionalClass, str, DispersionCorrection]:
    """Базовый выбор «функционал + базис + дисперсия» по профилю.

    Обоснование: гибридные функционалы с эмпирической дисперсией дают
    предсказуемую точность геометрий и энергий для органических молекул при
    умеренной стоимости; def2-семейство сбалансировано по элементам и не
    требует отдельных поляризационных добавок.
    """
    if profile is PrecisionProfile.SCREENING:
        return ("pbe", FunctionalClass.GGA, "def2-svp", DispersionCorrection.D3_BJ)
    if profile is PrecisionProfile.STANDARD:
        return ("pbe0", FunctionalClass.HYBRID, "def2-svp", DispersionCorrection.D3_BJ)
    if profile is PrecisionProfile.HIGH_ACCURACY:
        return ("pbe0", FunctionalClass.HYBRID, "def2-tzvp", DispersionCorrection.D3_BJ)
    return (
        "wb97x-d",
        FunctionalClass.RANGE_SEPARATED_HYBRID,
        "def2-tzvp",
        DispersionCorrection.NONE,
    )


def _numerics(profile: PrecisionProfile, task: Task, is_large: bool) -> tuple[GridPreset, ScfSpec]:
    """Сетка и пороги SCF.

    Частотные расчёты требуют более жёстких порогов и мелкой сетки: численное
    дифференцирование усиливает любой шум в энергии, и «дрожание» градиента
    превращается в ложные мнимые частоты.
    """
    thresholds: dict[PrecisionProfile, tuple[float, float]] = {
        PrecisionProfile.SCREENING: (1e-6, 1e-5),
        PrecisionProfile.STANDARD: (1e-8, 1e-6),
        PrecisionProfile.HIGH_ACCURACY: (1e-9, 1e-8),
        PrecisionProfile.RESEARCH: (1e-10, 1e-8),
    }
    grids: dict[PrecisionProfile, GridPreset] = {
        PrecisionProfile.SCREENING: GridPreset.FINE,
        PrecisionProfile.STANDARD: GridPreset.FINE,
        PrecisionProfile.HIGH_ACCURACY: GridPreset.ULTRAFINE,
        PrecisionProfile.RESEARCH: GridPreset.ULTRAFINE,
    }
    energy_threshold, density_threshold = thresholds[profile]
    grid = grids[profile]

    if task is Task.FREQUENCIES or task is Task.TS_OPTIMIZATION:
        energy_threshold = min(energy_threshold, 1e-10)
        density_threshold = min(density_threshold, 1e-8)
        grid = GridPreset.ULTRAFINE
    if is_large:
        grid = GridPreset.FINE

    scf = ScfSpec(
        max_iterations=80 if profile is PrecisionProfile.SCREENING else 128,
        energy_threshold=energy_threshold,
        density_threshold=density_threshold,
        stability_analysis=profile in (PrecisionProfile.HIGH_ACCURACY, PrecisionProfile.RESEARCH),
        damping=0.2 if profile is PrecisionProfile.RESEARCH else 0.0,
    )
    return grid, scf


def _integral_threshold(profile: PrecisionProfile, is_large: bool) -> float:
    """Порог отсечки двухэлектронных интегралов.

    Отсечка на уровне 1e-10 вносит ошибку энергии много меньшую порога
    сходимости SCF, поэтому для скрининга её можно ослабить до 1e-9, а для
    крупных систем — до 1e-8 ради скорости.
    """
    base = {
        PrecisionProfile.SCREENING: 1e-9,
        PrecisionProfile.STANDARD: 1e-10,
        PrecisionProfile.HIGH_ACCURACY: 1e-11,
        PrecisionProfile.RESEARCH: 1e-12,
    }[profile]
    return 1e-8 if is_large else base


def _optimization(task: Task) -> OptimizationSpec:
    """Параметры оптимизации: для поиска ТС шаг доверия уменьшают.

    В седловой точке квазиньютоновский шаг с большим радиусом доверия легко
    «перепрыгивает» через максимум, поэтому радиус урезан вдвое, а обновление
    гессиана заменено на Bofill — оно устойчивее при наличии отрицательной
    кривизны.
    """
    if task is Task.TS_OPTIMIZATION:
        return OptimizationSpec(
            coordinates=_COORDINATES,
            trust_radius=0.15,
            hessian_update="bofill",
            max_steps=150,
            max_force=3.0e-4,
            rms_force=2.0e-4,
        )
    return OptimizationSpec(coordinates=_COORDINATES)
