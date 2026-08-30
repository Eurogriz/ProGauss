"""Верификационный набор (§26 ТЗ, проект — docs/architecture/09).

Отвечает на вопрос «считаем ли мы правильно?» — в отличие от бенчмарков,
которые отвечают на вопрос «считаем ли мы быстро?».

Кейсы описаны данными в ``verification/cases/<уровень>/*.yaml``, поэтому новая
проверка добавляется без изменения кода. Неизвестное поле в кейсе — ошибка:
молча пропущенная проверка хуже её отсутствия, потому что создаёт видимость
покрытия.

Уровни, для которых кейсов нет (L3 гессиан, L4 частоты, L8 спин), отсутствуют
намеренно: соответствующие расчёты не реализованы, и фиктивные эталоны для них
появиться не должны (§54 ТЗ).

Запуск::

    python verification/run_verification.py            # весь набор
    python verification/run_verification.py --level L1 # один уровень
    python verification/run_verification.py --list     # только перечень
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from quantumlab.domain.molecule import Atom, Molecule  # noqa: E402
from quantumlab.domain.result import CalculationResult  # noqa: E402
from quantumlab.domain.spec import (  # noqa: E402
    CalculationSpec,
    GridSpec,
    MethodSpec,
    OptimizationSpec,
    SpinTreatment,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import BasisSet, build_basis  # noqa: E402
from quantumlab.engine.constants import angstrom_to_bohr  # noqa: E402
from quantumlab.engine.contracts import EngineRequest  # noqa: E402
from quantumlab.engine.gradients import (  # noqa: E402
    rhf_gradient,
    rohf_gradient,
    uhf_gradient,
)
from quantumlab.engine.reference import ReferenceEngine  # noqa: E402
from quantumlab.engine.scf import run_rhf, run_rohf, run_uhf  # noqa: E402

CASES_DIR = Path(__file__).resolve().parent / "cases"


# --------------------------------------------------------------------------- #
# Схема кейса
# --------------------------------------------------------------------------- #
class Expectation(BaseModel):
    """Ожидаемое численное значение с допуском и указанием источника.

    ``source`` обязателен: эталон без источника непроверяем, а допуск без
    источника невозможно обосновать (§26 ТЗ).
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    tol: float = Field(gt=0.0)
    source: str


class GeometryProbe(BaseModel):
    """Межъядерное расстояние или валентный угол в итоговой геометрии."""

    model_config = ConfigDict(extra="forbid")

    atoms: tuple[int, ...]
    value: float
    tol: float = Field(gt=0.0)
    source: str = "internal-reference"


class FiniteDifferenceCheck(BaseModel):
    """Сверка аналитического градиента с численной производной энергии."""

    model_config = ConfigDict(extra="forbid")

    step_angstrom: float = 1.0e-4
    max_deviation_eh_bohr: float = Field(default=1.0e-6, gt=0.0)


class InvarianceCheck(BaseModel):
    """Инвариантность энергии к вращению системы и к перенумерации атомов."""

    model_config = ConfigDict(extra="forbid")

    max_energy_deviation_eh: float = Field(default=1.0e-10, gt=0.0)


class ReproducibilityCheck(BaseModel):
    """Повтор того же расчёта обязан дать тот же результат."""

    model_config = ConfigDict(extra="forbid")

    max_energy_deviation_eh: float = Field(default=0.0, ge=0.0)


class Case(BaseModel):
    """Один верификационный кейс."""

    model_config = ConfigDict(extra="forbid")

    id: str
    level: str
    description: str = ""
    slow: bool = Field(
        default=False,
        description=(
            "Кейс требует десятков SCF-расчётов и не входит в быстрый прогон. "
            "Ставится по измеренной стоимости, а не на глаз."
        ),
    )
    molecule: str
    charge: int = Field(
        default=0, description="Суммарный заряд системы; нужен для открытых оболочек."
    )
    multiplicity: int = Field(default=1, ge=1, description="Спиновая мультиплетность 2S+1.")
    spec: dict[str, Any]
    expect: dict[str, Expectation] = Field(default_factory=dict)
    bond_lengths_angstrom: tuple[GeometryProbe, ...] = ()
    angles_degrees: tuple[GeometryProbe, ...] = ()
    gradient_vs_finite_difference: FiniteDifferenceCheck | None = None
    invariance: InvarianceCheck | None = None
    reproducibility: ReproducibilityCheck | None = None
    quality: dict[str, bool] = Field(default_factory=dict)


@dataclass(slots=True)
class Outcome:
    """Итог проверки одного кейса."""

    case_id: str
    level: str
    passed: bool
    lines: list[str] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- #
# Запуск расчёта
# --------------------------------------------------------------------------- #
def _build_spec(raw: dict[str, Any]) -> CalculationSpec:
    method_raw = dict(raw["method"])
    theory = TheoryFamily(str(method_raw.pop("theory")).lower())
    method = MethodSpec(theory=theory, **method_raw)
    optimization = OptimizationSpec(**raw.get("optimization", {}))
    # Сетка передаётся явно: энергия DFT без указания пресета сетки — число
    # без определённого значения, два прогона на разных пресетах различаются.
    grid = GridSpec(**raw.get("grid", {}))
    return CalculationSpec(
        task=Task(raw["task"]),
        method=method,
        optimization=optimization,
        grid=grid,
    )


def _load_molecule(case: Case) -> Molecule:
    path = REPO_ROOT / case.molecule
    return Molecule.from_xyz(
        path.read_text(encoding="utf-8"),
        name=path.stem,
        charge=case.charge,
        multiplicity=case.multiplicity,
    )


def _run(case: Case, molecule: Molecule | None = None) -> CalculationResult:
    engine = ReferenceEngine()
    request = EngineRequest(
        job_id=f"verification-{case.id}",
        molecule=molecule if molecule is not None else _load_molecule(case),
        spec=_build_spec(case.spec),
    )
    return engine.run(request)


# --------------------------------------------------------------------------- #
# Отдельные виды проверок
# --------------------------------------------------------------------------- #
def _rotated(molecule: Molecule) -> Molecule:
    """Поворот на 37° вокруг оси (1, 1, 1) — заведомо не вдоль осей координат."""
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    angle = 37.0 * np.pi / 180.0
    cosine, sine = np.cos(angle), np.sin(angle)
    outer = np.outer(axis, axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    matrix = cosine * np.eye(3) + sine * cross + (1.0 - cosine) * outer
    atoms = tuple(
        atom.model_copy(
            update={"position": tuple(float(v) for v in matrix @ np.array(atom.position))}
        )
        for atom in molecule.atoms
    )
    return molecule.model_copy(update={"atoms": atoms})


def _permuted(molecule: Molecule) -> Molecule:
    """Обратный порядок атомов: проверяет независимость от нумерации."""
    return molecule.model_copy(update={"atoms": tuple(reversed(molecule.atoms))})


def _resolve_spin(spin: SpinTreatment | str) -> SpinTreatment:
    return SpinTreatment(str(spin).lower()) if isinstance(spin, str) else spin


def _spin_energy(basis: BasisSet, molecule: Molecule, spin: SpinTreatment) -> float:
    """Энергия SCF с учётом обработки спина.

    Сверка конечными разностями проверяет ту же поверхность, что и
    аналитический градиент, а не RHF-поверхность.
    """
    if spin is SpinTreatment.UHF:
        return float(run_uhf(basis, molecule).total_energy)
    if spin is SpinTreatment.ROHF:
        return float(run_rohf(basis, molecule).total_energy)
    return float(run_rhf(basis, molecule).total_energy)


def _spin_gradient(basis: BasisSet, molecule: Molecule, spin: SpinTreatment) -> np.ndarray:
    """Аналитический градиент с учётом обработки спина."""
    if spin is SpinTreatment.UHF:
        return uhf_gradient(basis, molecule, run_uhf(basis, molecule)).gradient
    if spin is SpinTreatment.ROHF:
        return rohf_gradient(basis, molecule, run_rohf(basis, molecule)).gradient
    return rhf_gradient(basis, molecule, run_rhf(basis, molecule)).gradient


def _displaced_energy(
    molecule: Molecule,
    basis_name: str,
    atom_index: int,
    axis: int,
    step: float,
    spin: SpinTreatment = SpinTreatment.RHF,
) -> float:
    atoms = list(molecule.atoms)
    x, y, z = atoms[atom_index].position
    coordinates = [x, y, z]
    coordinates[axis] += step
    atoms[atom_index] = Atom(
        symbol=atoms[atom_index].symbol,
        position=(coordinates[0], coordinates[1], coordinates[2]),
    )
    moved = molecule.model_copy(update={"atoms": tuple(atoms)})
    basis = build_basis(basis_name, moved)
    return _spin_energy(basis, moved, spin)


def _numerical_gradient(
    molecule: Molecule,
    basis_name: str,
    step: float,
    spin: SpinTreatment = SpinTreatment.RHF,
) -> np.ndarray:
    """Центральная разность энергии по декартовым координатам ядер.

    Делитель переводит шаг из ангстрем в бор, потому что градиент ядро
    возвращает в э/бор.
    """
    divisor = 2.0 * angstrom_to_bohr(step)
    gradient = np.zeros((len(molecule.atoms), 3))
    for index in range(len(molecule.atoms)):
        for axis in range(3):
            forward = _displaced_energy(molecule, basis_name, index, axis, +step, spin)
            backward = _displaced_energy(molecule, basis_name, index, axis, -step, spin)
            gradient[index, axis] = (forward - backward) / divisor
    return gradient


def _bond_length(molecule: Molecule, first: int, second: int) -> float:
    a = np.array(molecule.atoms[first].position)
    b = np.array(molecule.atoms[second].position)
    return float(np.linalg.norm(a - b))


def _angle(molecule: Molecule, first: int, vertex: int, second: int) -> float:
    a = np.array(molecule.atoms[first].position) - np.array(molecule.atoms[vertex].position)
    b = np.array(molecule.atoms[second].position) - np.array(molecule.atoms[vertex].position)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


# --------------------------------------------------------------------------- #
# Выполнение кейса
# --------------------------------------------------------------------------- #
def _check(case: Case) -> Outcome:
    outcome = Outcome(case_id=case.id, level=case.level, passed=True)
    molecule = _load_molecule(case)
    basis_name = str(case.spec["method"]["basis"])
    result = _run(case, molecule)

    for name, expectation in case.expect.items():
        obtained = _field(result, name)
        deviation = abs(obtained - expectation.value)
        ok = deviation <= expectation.tol
        outcome.passed &= ok
        outcome.lines.append(
            f"    {name}: {obtained:.10f} (эталон {expectation.value:.10f} ±{expectation.tol:g}, "
            f"|Δ| = {deviation:.3e}, источник: {expectation.source}) {'✓' if ok else '✗'}"
        )

    geometry = result.final_molecule
    for probe in case.bond_lengths_angstrom:
        if geometry is None:
            outcome.passed = False
            outcome.lines.append("    длина связи: нет итоговой геометрии ✗")
            continue
        first, second = probe.atoms
        obtained = _bond_length(geometry, first, second)
        deviation = abs(obtained - probe.value)
        ok = deviation <= probe.tol
        outcome.passed &= ok
        outcome.lines.append(
            f"    r({first}-{second}): {obtained:.6f} Å (эталон {probe.value:.6f} "
            f"±{probe.tol:g}, |Δ| = {deviation:.3e}) {'✓' if ok else '✗'}"
        )

    for probe in case.angles_degrees:
        if geometry is None:
            outcome.passed = False
            outcome.lines.append("    валентный угол: нет итоговой геометрии ✗")
            continue
        first, vertex, second = probe.atoms
        obtained = _angle(geometry, first, vertex, second)
        deviation = abs(obtained - probe.value)
        ok = deviation <= probe.tol
        outcome.passed &= ok
        outcome.lines.append(
            f"    ∠({first}-{vertex}-{second}): {obtained:.4f}° (эталон {probe.value:.4f} "
            f"±{probe.tol:g}, |Δ| = {deviation:.3e}) {'✓' if ok else '✗'}"
        )

    if case.gradient_vs_finite_difference is not None:
        specification = case.gradient_vs_finite_difference
        spin = _resolve_spin(case.spec["method"].get("spin", "rhf"))
        analytical = _spin_gradient(build_basis(basis_name, molecule), molecule, spin)
        numerical = _numerical_gradient(molecule, basis_name, specification.step_angstrom, spin)
        deviation = float(np.abs(analytical - numerical).max())
        ok = deviation <= specification.max_deviation_eh_bohr
        outcome.passed &= ok
        outcome.lines.append(
            f"    градиент против конечных разностей (h = {specification.step_angstrom:g} Å): "
            f"max|Δ| = {deviation:.3e} э/бор (допуск {specification.max_deviation_eh_bohr:g}) "
            f"{'✓' if ok else '✗'}"
        )

    if case.invariance is not None:
        limit = case.invariance.max_energy_deviation_eh
        reference_energy = result.energy_hartree
        for label, variant in (
            ("вращение", _rotated(molecule)),
            ("перенумерация", _permuted(molecule)),
        ):
            moved = _run(case, variant).energy_hartree
            deviation = abs(moved - reference_energy)
            ok = deviation <= limit
            outcome.passed &= ok
            outcome.lines.append(
                f"    инвариантность к «{label}»: |ΔE| = {deviation:.3e} Eh "
                f"(допуск {limit:g}) {'✓' if ok else '✗'}"
            )

    if case.reproducibility is not None:
        limit = case.reproducibility.max_energy_deviation_eh
        repeated = _run(case, molecule)
        deviation = abs(repeated.energy_hartree - result.energy_hartree)
        same_fingerprint = repeated.fingerprint == result.fingerprint
        ok = deviation <= limit and same_fingerprint
        outcome.passed &= ok
        outcome.lines.append(
            f"    повтор расчёта: |ΔE| = {deviation:.3e} Eh, отпечаток совпал: "
            f"{same_fingerprint} (допуск {limit:g}) {'✓' if ok else '✗'}"
        )

    for name, required in case.quality.items():
        check = result.checks_by_name().get(name)
        verdict = None if check is None else check.verdict.value
        ok = (verdict == "pass") is required
        outcome.passed &= ok
        outcome.lines.append(
            f"    проверка качества «{name}»: {verdict or 'отсутствует'} "
            f"(требуется {'pass' if required else 'не pass'}) {'✓' if ok else '✗'}"
        )

    if not outcome.lines:
        outcome.passed = False
        outcome.lines.append("    в кейсе нет ни одной проверки — это ошибка оформления ✗")
    return outcome


def _field(result: CalculationResult, name: str) -> float:
    if not hasattr(result, name):
        msg = f"поле «{name}» отсутствует в CalculationResult"
        raise KeyError(msg)
    value = getattr(result, name)
    if value is None:
        msg = f"поле «{name}» пусто для этого расчёта"
        raise ValueError(msg)
    return float(value)


# --------------------------------------------------------------------------- #
# Точка входа
# --------------------------------------------------------------------------- #
def load_cases(levels: tuple[str, ...] = (), *, include_slow: bool = True) -> list[Case]:
    """Читает кейсы; ошибка схемы или неизвестное поле прерывают загрузку."""
    cases: list[Case] = []
    for path in sorted(CASES_DIR.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            case = Case.model_validate(raw)
        except ValidationError as error:
            msg = f"некорректный кейс {path.relative_to(REPO_ROOT)}:\n{error}"
            raise ValueError(msg) from error
        if levels and case.level not in levels:
            continue
        if case.slow and not include_slow:
            continue
        cases.append(case)
    if not cases:
        msg = "не найдено ни одного кейса — каталог verification/cases пуст?"
        raise ValueError(msg)
    return cases


def run(
    levels: tuple[str, ...] = (),
    only: tuple[str, ...] = (),
    *,
    include_slow: bool = True,
) -> list[Outcome]:
    """Выполняет кейсы и печатает абсолютные числа по каждому."""
    outcomes: list[Outcome] = []
    for case in load_cases(levels, include_slow=include_slow):
        if only and case.id not in only:
            continue
        print(f"{case.level}  {case.id}")
        if case.description:
            print(f"    {case.description}")
        try:
            outcome = _check(case)
        except Exception as error:  # кейс обязан давать диагноз, а не ронять весь набор
            outcome = Outcome(
                case_id=case.id,
                level=case.level,
                passed=False,
                error=f"{type(error).__name__}: {error}",
            )
        for line in outcome.lines:
            print(line)
        if outcome.error is not None:
            print(f"    ОШИБКА: {outcome.error}")
        print(f"    → {'ПРОЙДЕНО' if outcome.passed else 'ПРОВАЛ'}")
        outcomes.append(outcome)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI: 0 при полном прохождении, 1 при любом провале."""
    parser = argparse.ArgumentParser(description="Верификационный набор QuantumLab (§26 ТЗ)")
    parser.add_argument(
        "--level", action="append", default=[], help="только указанный уровень (L1…L9)"
    )
    parser.add_argument("--case", action="append", default=[], help="только указанный id кейса")
    parser.add_argument("--list", action="store_true", help="перечислить кейсы и выйти")
    parser.add_argument("--json", action="store_true", help="вывести итог в JSON")
    parser.add_argument("--fast", action="store_true", help="пропустить кейсы с флагом slow")
    arguments = parser.parse_args(argv)

    if arguments.list:
        for case in load_cases(tuple(arguments.level)):
            print(f"{case.level}  {case.id}  ({case.molecule})")
        return 0

    outcomes = run(tuple(arguments.level), tuple(arguments.case), include_slow=not arguments.fast)
    failed = [item for item in outcomes if not item.passed]

    if arguments.json:
        print(
            json.dumps(
                {
                    "total": len(outcomes),
                    "failed": [item.case_id for item in failed],
                    "by_level": sorted({item.level for item in outcomes}),
                },
                ensure_ascii=False,
            )
        )

    print()
    print(f"Кейсов: {len(outcomes)}, провалено: {len(failed)}")
    for item in failed:
        print(f"  ✗ {item.case_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
