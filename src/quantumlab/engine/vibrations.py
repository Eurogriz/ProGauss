"""Колебательный анализ: численный гессиан из аналитических градиентов.

Частоты считаются не аналитическим гессианом (его нет ни для одного метода), а
центральными разностями **аналитического градиента**. Это принципиально лучше
разностей энергии: при том же шаге ошибка на порядок меньше, потому что
аналитический градиент сам уже на порядок точнее численного.

Что здесь физика, а что численный метод:

* масс-взвешивание и проекция поступательных/вращательных мод — точные
  соотношения механики, ошибок дискретизации в них нет;
* переход от собственных значений к см⁻¹ — точное преобразование единиц;
* сам гессиан — единственный численный элемент, и его шаг виден в результате:
  при слишком большом шаге растёт ошибка отбрасывания, при слишком малом —
  ошибка округления.

Отрицательные собственные значения возвращаются как **мнимые частоты** с
отрицательным значением в см⁻¹, а не отбрасываются: одна мнимая частота — это
признак переходного состояния, и скрывать его означало бы выдать седловую точку
за минимум (§54 ТЗ).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quantumlab.domain.molecule import Atom, Molecule, element_from_symbol
from quantumlab.engine.constants import (
    AMU_TO_ELECTRON_MASS,
    ANGSTROM_TO_BOHR,
    HARTREE_TO_CM1,
)

Array = npt.NDArray[np.float64]

#: Шаг центральных разностей в борах. Для гессиана из аналитического градиента
#: оптимум заметно крупнее, чем для градиента из энергии: градиент вычисляется
#: почти точно, поэтому ошибка округления не растёт так быстро.
DEFAULT_HESSIAN_STEP_BOHR: float = 1e-3


@dataclass(frozen=True)
class Vibrations:
    """Результат колебательного анализа.

    Attributes:
        frequencies_cm1: частоты в см⁻¹ по возрастанию. Отрицательные значения —
            мнимые частоты; их ровно столько, сколько отрицательных собственных
            значений у масс-взвешенного гессиана.
        zero_point_energy_hartree: ``½ Σ ħω`` по действительным модам.
        imaginary_frequencies: только мнимые частоты — по ним видно, минимум это
            или седловая точка.
        rigid_modes: сколько поступательно-вращательных мод спроецировано
            (6 для нелинейной молекулы, 5 для линейной).
    """

    frequencies_cm1: tuple[float, ...]
    zero_point_energy_hartree: float
    imaginary_frequencies: tuple[float, ...]
    rigid_modes: int


def _displace(molecule: Molecule, atom_index: int, axis: int, delta_bohr: float) -> Molecule:
    """Смещает одну координату. Заряд и мультиплетность обязаны сохраниться."""
    delta_angstrom = delta_bohr / ANGSTROM_TO_BOHR
    atoms: list[Atom] = []
    for index, atom in enumerate(molecule.atoms):
        position = list(atom.position)
        if index == atom_index:
            position[axis] += delta_angstrom
        atoms.append(Atom(symbol=atom.symbol, position=(position[0], position[1], position[2])))
    return Molecule(
        atoms=tuple(atoms),
        name=molecule.name,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )


def atomic_masses_au(molecule: Molecule) -> Array:
    """Массы атомов в массах электрона — единица массы атомной системы."""
    return np.array(
        [
            element_from_symbol(atom.symbol).mass_amu * AMU_TO_ELECTRON_MASS
            for atom in molecule.atoms
        ]
    )


def numerical_hessian(
    molecule: Molecule,
    gradient: Callable[[Molecule], Array],
    *,
    step_bohr: float = DEFAULT_HESSIAN_STEP_BOHR,
) -> Array:
    """Гессиан ``3N × 3N`` центральными разностями аналитического градиента.

    ``gradient`` возвращает силы в хартри/бор, поэтому шаг задаётся в борах, а
    координаты в молекуле — в ангстремах: перевод выполняется здесь, чтобы
    вызывающая сторона не держала в голове две системы единиц.

    Результат симметризуется: аналитически гессиан симметричен, а численно —
    лишь с точностью до погрешности разностей. Симметризация стоит ничего и
    убирает асимметрию, из-за которой ``eigvalsh`` дал бы другой спектр.
    """
    size = 3 * len(molecule.atoms)
    hessian = np.zeros((size, size))
    for index in range(size):
        atom_index, axis = divmod(index, 3)
        plus = gradient(_displace(molecule, atom_index, axis, step_bohr))
        minus = gradient(_displace(molecule, atom_index, axis, -step_bohr))
        hessian[:, index] = (plus - minus).reshape(size) / (2 * step_bohr)
    return 0.5 * (hessian + hessian.T)


def rigid_body_modes(molecule: Molecule) -> Array:
    """Ортонормированный базис поступательных и вращательных смещений.

    Векторы строятся сразу в масс-взвешенных координатах ``q = √m·x``, где
    метрика евклидова: перенос всего тела на ``δ e_k`` даёт ``√m_A δ_{ak}``,
    а вращение с угловой скоростью вокруг оси ``k`` — ``√m_A (e_k × r_A)`` с
    ``r_A`` от центра масс.

    Для линейной молекулы одно из вращений вырождается (вектор зануляется), и
    после ортогонализации остаётся 5 базисных векторов вместо 6. Отбрасывать
    его по флагу «линейна ли молекула» нельзя: порог надёжнее классификации,
    которая сама зависит от допуска.
    """
    positions = np.array(
        [[ANGSTROM_TO_BOHR * value for value in atom.position] for atom in molecule.atoms]
    )
    masses = atomic_masses_au(molecule)
    center = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    relative = positions - center
    sqrt_masses = np.sqrt(masses)

    raw: list[Array] = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = sqrt_masses
        raw.append(translation.reshape(-1))
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 1.0
        rotation = np.cross(np.broadcast_to(direction, relative.shape), relative)
        raw.append((sqrt_masses[:, None] * rotation).reshape(-1))

    basis: list[Array] = []
    for vector in raw:
        residual = vector.copy()
        for existing in basis:
            residual = residual - float(residual @ existing) * existing
        norm = float(np.linalg.norm(residual))
        if norm > 1e-8:
            basis.append(residual / norm)
    return np.stack(basis)


def vibrational_analysis(hessian: Array, molecule: Molecule) -> Vibrations:
    """Частоты и нулевая энергия из гессиана.

    Масс-взвешенный гессиан ``M^{-1/2} H M^{-1/2}`` проецируется на дополнение к
    подпространству переносов и вращений, и уже там диагонализуется. Без
    проекции шесть нулевых собственных значений смешались бы с колебательными и
    дали бы мнимые частоты там, где их физически нет.

    ``ν̃ = ω/(2πc)``, а в атомных единицах ``ħ = 1``, поэтому ``ω`` численно
    равно энергии. Делить ещё и на ``2π`` здесь не нужно: ``HARTREE_TO_CM1``
    — это ``E_h/(hc)``, то есть ``h = 2πħ`` уже учтено. Итого перевод —
    просто умножение на ``HARTREE_TO_CM1``.
    """
    size = 3 * len(molecule.atoms)
    masses = atomic_masses_au(molecule)
    inverse_sqrt = np.repeat(1.0 / np.sqrt(masses), 3)
    weighted = hessian * np.outer(inverse_sqrt, inverse_sqrt)

    rigid = rigid_body_modes(molecule)
    projector = np.eye(size) - rigid.T @ rigid
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projector + projector.T))
    complement = eigenvectors[:, eigenvalues > 0.5]

    reduced = complement.T @ weighted @ complement
    spectrum = np.linalg.eigvalsh(0.5 * (reduced + reduced.T))

    signed = np.sign(spectrum) * np.sqrt(np.abs(spectrum))
    frequencies = tuple(float(value * HARTREE_TO_CM1) for value in signed)
    real = [value for value in signed if value > 0.0]
    imaginary = tuple(frequency for frequency in frequencies if frequency < 0.0)
    return Vibrations(
        frequencies_cm1=frequencies,
        zero_point_energy_hartree=0.5 * float(sum(real)),
        imaginary_frequencies=imaginary,
        rigid_modes=rigid.shape[0],
    )
