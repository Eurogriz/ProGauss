"""Оптимизация геометрии: сходимость, замороженные атомы, честный отказ.

Эталон для проверки — не «правдоподобная длина связи», а минимум той же самой
энергетической поверхности, найденный независимым способом (бисекцией по
градиенту). Оптимизатор обязан привести в ту же точку, откуда бы он ни стартовал.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.engine.basis import build_basis
from quantumlab.engine.gradients import rhf_gradient
from quantumlab.engine.optimizer import (
    OptimizationResult,
    OptimizationSettings,
    optimize_geometry,
)
from quantumlab.engine.scf import ScfSettings, run_rhf

BASIS = "sto-3g"


def _h2(distance: float) -> Molecule:
    return Molecule(
        name="h2",
        atoms=(
            Atom(symbol="H", position=(0.0, 0.0, 0.0)),
            Atom(symbol="H", position=(0.0, 0.0, distance)),
        ),
    )


def _energy_and_gradient(molecule: Molecule) -> tuple[float, np.ndarray]:
    basis = build_basis(BASIS, molecule)
    scf = run_rhf(basis, molecule, ScfSettings())
    if not scf.converged:
        msg = "SCF не сошёлся — оптимизация по такой плотности недопустима"
        raise RuntimeError(msg)
    return scf.total_energy, rhf_gradient(basis, molecule, scf).gradient


def _bond_length(molecule: Molecule) -> float:
    first = np.array(molecule.atoms[0].position)
    second = np.array(molecule.atoms[1].position)
    return float(np.linalg.norm(second - first))


def _minimum_by_bisection(low: float = 0.5, high: float = 1.2) -> float:
    """Независимый поиск минимума: бисекция по знаку производной.

    Нужен как эталон: сравнивать оптимизатор с ним — значит проверять его
    другим методом, а не тем же самым.
    """

    def slope(distance: float) -> float:
        molecule = _h2(distance)
        return float(
            rhf_gradient(
                build_basis(BASIS, molecule),
                molecule,
                run_rhf(build_basis(BASIS, molecule), molecule, ScfSettings()),
            ).gradient[1, 2]
        )

    for _ in range(40):
        middle = (low + high) / 2.0
        if slope(middle) < 0.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# --------------------------------------------------------------------------- #
# Сходимость
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("start", [0.60, 0.7414, 0.95])
def test_converges_to_the_same_geometry_from_any_start(start: float) -> None:
    """Сжатая, равновесная и растянутая стартовые точки дают одну геометрию."""
    result = optimize_geometry(_h2(start), _energy_and_gradient, OptimizationSettings(max_steps=30))
    assert result.converged
    assert result.reason_key == "optimization.converged"
    assert _bond_length(result.molecule) == pytest.approx(_minimum_by_bisection(), abs=1e-3)


def test_energy_never_increases() -> None:
    """Монотонность: каждый принятый шаг понижает энергию (или не меняет её)."""
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, OptimizationSettings(max_steps=30))
    energies = [step.energy_hartree for step in result.history]
    for previous, current in pairwise(energies):
        assert current <= previous + 1e-12


def test_energy_at_the_result_matches_the_returned_geometry() -> None:
    """Энергия в результате обязана относиться к возвращённой геометрии.

    Регрессия: рассогласование «энергия из одной точки, геометрия из другой»
    выглядит правдоподобно, но делает результат бессмысленным.
    """
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, OptimizationSettings(max_steps=30))
    energy, _ = _energy_and_gradient(result.molecule)
    assert energy == pytest.approx(result.energy_hartree, rel=1e-10)


def test_final_gradient_is_below_the_thresholds() -> None:
    """Сошедшаяся оптимизация заканчивается с силой ниже порогов спецификации."""
    settings = OptimizationSettings(max_steps=30)
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, settings)
    assert result.converged
    assert result.max_force < settings.max_force
    last = result.history[-1]
    assert last.rms_force is not None
    assert last.rms_force < settings.rms_force


def test_history_records_every_iteration() -> None:
    """Журнал нужен UI для графика сходимости — пропусков быть не должно."""
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, OptimizationSettings(max_steps=30))
    assert [step.index for step in result.history] == list(range(result.steps + 1))
    assert result.history[0].max_displacement is None  # на первом шаге смещения нет
    assert result.history[1].max_displacement is not None


# --------------------------------------------------------------------------- #
# Ограничения и отказы
# --------------------------------------------------------------------------- #
def test_frozen_atoms_do_not_move() -> None:
    """Замороженный атом остаётся на месте, остальная молекула подстраивается."""
    start = _h2(0.95)
    settings = OptimizationSettings(max_steps=30, frozen_atoms=(0,))
    result = optimize_geometry(start, _energy_and_gradient, settings)
    assert result.molecule.atoms[0].position == start.atoms[0].position
    assert _bond_length(result.molecule) != pytest.approx(0.95, abs=1e-3)


def test_all_frozen_atoms_mean_no_movement() -> None:
    """Если заморожено всё, геометрия не меняется — и это не «сходимость»."""
    start = _h2(0.95)
    result = optimize_geometry(
        start, _energy_and_gradient, OptimizationSettings(max_steps=5, frozen_atoms=(0, 1))
    )
    assert _bond_length(result.molecule) == pytest.approx(0.95)
    assert not result.converged


def test_unknown_frozen_atom_is_rejected() -> None:
    """Индекс несуществующего атома — явная ошибка, а не тихий пропуск."""
    with pytest.raises(IndexError, match="не существует"):
        optimize_geometry(_h2(0.95), _energy_and_gradient, OptimizationSettings(frozen_atoms=(7,)))


def test_exhausted_steps_are_reported_honestly() -> None:
    """Исчерпание шагов — результат с причиной, а не исключение.

    Пользователь обязан увидеть, что оптимизация не сошлась, вместе с
    последней геометрией: выбрасывать её было бы потерей работы.
    """
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, OptimizationSettings(max_steps=1))
    assert not result.converged
    assert result.reason_key == "optimization.max_steps_reached"
    assert result.steps == 1
    assert result.max_force > OptimizationSettings().max_force


def test_trust_radius_limits_the_step() -> None:
    """Доверительный радиус действительно ограничивает длину шага."""
    settings = OptimizationSettings(max_steps=2, trust_radius=0.01)
    result = optimize_geometry(_h2(0.95), _energy_and_gradient, settings)
    for step in result.history[1:]:
        assert step.max_displacement is not None
        # max_displacement — наибольшая компонента, она не превышает нормы шага.
        assert step.max_displacement <= settings.trust_radius + 1e-12


def test_result_type_is_exported() -> None:
    """Тип результата доступен для аннотаций в вызывающем коде."""
    result = optimize_geometry(_h2(0.7414), _energy_and_gradient, OptimizationSettings(max_steps=3))
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.gradient, np.ndarray)
    assert result.gradient.shape == (2, 3)
