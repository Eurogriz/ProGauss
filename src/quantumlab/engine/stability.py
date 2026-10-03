"""Анализ устойчивости хартри-фоковской волновой функции.

SCF сходится к *стационарной* точке энергии по вращениям орбиталей, но не
обязательно к минимуму: седловая точка тоже даёт ``FDS = SDF`` и тот же
критерий сходимости. Различить их можно по собственным значениям орбитального
гессиана. Отрицательное значение означает, что существует решение с меньшей
энергией, а число из текущего расчёта — не основное состояние (§54 ТЗ).

Реализованы вещественные устойчивости, для которых гессиан — ``A + B``:

* **RHF → RHF** (синглетная, внутренняя): вращения, сохраняющие спиновую
  симметрию;
* **RHF → UHF** (триплетная, внешняя): нарушение ``α = β``. Именно она
  вскрывает растянутую связь, где RHF-решение — седловая точка;
* **UHF → UHF** (внутренняя, оба спиновых канала вместе).

С ``(ia|jb)`` — молекулярные ERI, ``i, j`` — занятые, ``a, b`` — виртуальные:

.. math::

    (A+B)^{S}_{ia,jb} = δ_{ij}δ_{ab}(ε_a−ε_i) + 4(ia|jb) − (ib|ja) − (ij|ab)

    (A+B)^{T}_{ia,jb} = δ_{ij}δ_{ab}(ε_a−ε_i) − (ib|ja) − (ij|ab)

    (A+B)^{σσ'}_{ia,jb} = δ_{σσ'}[δ_{ij}δ_{ab}(ε_a−ε_i) − (ib|ja) − (ij|ab)] + 2(ia|jb)

Собственные значения здесь — в хартри, в тех же единицах, что и
``ε_a − ε_i``. Комплексные вращения (``A − B``), ROHF и DFT (нужно XC-ядро
второго порядка) не реализованы и отклоняются на уровне движка.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Порог устойчивости, хартри. Совпадает со значением PySCF по умолчанию
#: (``tol=1e-4``): отрицательные значения в пределах численного шума
#: вырожденных вращений нестабильностью не считаются.
STABILITY_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class StabilityChannel:
    """Результат по одному типу вращений.

    Attributes:
        name: ``rhf->rhf``, ``rhf->uhf`` или ``uhf->uhf``.
        lowest_eigenvalue: наименьшее собственное значение гессиана, хартри.
        stable: ``lowest_eigenvalue ≥ −STABILITY_TOLERANCE``.
    """

    name: str
    lowest_eigenvalue: float
    stable: bool


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """Итог анализа устойчивости."""

    channels: tuple[StabilityChannel, ...]

    @property
    def stable(self) -> bool:
        """Устойчиво по всем проверенным типам вращений."""
        return all(channel.stable for channel in self.channels)

    @property
    def unstable_channels(self) -> tuple[StabilityChannel, ...]:
        """Каналы с отрицательным собственным значением."""
        return tuple(channel for channel in self.channels if not channel.stable)


def _coulomb_block(
    eri: np.ndarray, left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """Блок ``(ia|jb)`` с составными индексами: ``(i, a)`` — спин ``σ``, ``(j, b)`` — ``σ'``.

    ``left`` и ``right`` — пары ``(занятые, виртуальные)`` коэффициентов. ``(ia|jb)
    = Σ C_μi C_νa C_λj C_σb (μν|λσ)``.
    """
    occ_l, vir_l = left
    occ_r, vir_r = right
    half = np.einsum("mnls,mi,nv->ivls", eri, occ_l, vir_l, optimize=True)
    block = np.einsum("ivls,lj,sw->ivjw", half, occ_r, vir_r, optimize=True)
    shape = (occ_l.shape[1] * vir_l.shape[1], occ_r.shape[1] * vir_r.shape[1])
    return np.asarray(block.reshape(shape))


def _exchange_blocks(
    eri: np.ndarray, occ: np.ndarray, vir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Односпиновые блоки ``(ib|ja)`` и ``(ij|ab)`` с составными индексами ``ia, jb``."""
    n_o, n_v = occ.shape[1], vir.shape[1]
    # (ib|ja) = Σ C_μi C_νb C_λj C_σa (μν|λσ)
    half = np.einsum("mnls,mi,nb->ibls", eri, occ, vir, optimize=True)
    ibja = np.einsum("ibls,lj,sa->iajb", half, occ, vir, optimize=True)
    # (ij|ab)
    half = np.einsum("mnls,mi,nj->ijls", eri, occ, occ, optimize=True)
    ijab = np.einsum("ijls,la,sb->iajb", half, vir, vir, optimize=True)
    return ibja.reshape(n_o * n_v, n_o * n_v), ijab.reshape(n_o * n_v, n_o * n_v)


def _delta_epsilon(energies: np.ndarray, n_occ: int) -> np.ndarray:
    """Диагональ ``ε_a − ε_i`` для составного индекса ``ia``."""
    occupied = energies[:n_occ]
    virtual = energies[n_occ:]
    return np.asarray((virtual[None, :] - occupied[:, None]).reshape(-1))


def _lowest(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return float("inf")
    symmetric = 0.5 * (matrix + matrix.T)
    return float(np.linalg.eigvalsh(symmetric)[0])


def rhf_stability(
    coefficients: np.ndarray,
    orbital_energies: np.ndarray | tuple[float, ...],
    eri: np.ndarray,
    n_occupied: int,
) -> StabilityResult:
    """Устойчивость RHF: внутренняя (RHF→RHF) и внешняя триплетная (RHF→UHF)."""
    energies = np.asarray(orbital_energies, dtype=float)
    occ, vir = coefficients[:, :n_occupied], coefficients[:, n_occupied:]
    if vir.shape[1] == 0 or n_occupied == 0:
        return StabilityResult(channels=())
    coulomb = _coulomb_block(eri, (occ, vir), (occ, vir))
    ibja, ijab = _exchange_blocks(eri, occ, vir)
    diagonal = np.diag(_delta_epsilon(energies, n_occupied))
    singlet = diagonal + 4.0 * coulomb - ibja - ijab
    triplet = diagonal - ibja - ijab
    return StabilityResult(
        channels=(
            _channel("rhf->rhf", _lowest(singlet)),
            _channel("rhf->uhf", _lowest(triplet)),
        )
    )


def uhf_stability(
    alpha_coefficients: np.ndarray,
    beta_coefficients: np.ndarray,
    alpha_energies: np.ndarray | tuple[float, ...],
    beta_energies: np.ndarray | tuple[float, ...],
    eri: np.ndarray,
    n_alpha: int,
    n_beta: int,
) -> StabilityResult:
    """Внутренняя устойчивость UHF: вращения α и β рассматриваются вместе."""
    e_alpha = np.asarray(alpha_energies, dtype=float)
    e_beta = np.asarray(beta_energies, dtype=float)
    n_va = alpha_coefficients.shape[1] - n_alpha
    n_vb = beta_coefficients.shape[1] - n_beta
    sizes = (n_alpha * n_va, n_beta * n_vb)
    if sum(sizes) == 0:
        return StabilityResult(channels=())

    occ_a, vir_a = alpha_coefficients[:, :n_alpha], alpha_coefficients[:, n_alpha:]
    occ_b, vir_b = beta_coefficients[:, :n_beta], beta_coefficients[:, n_beta:]
    aa = _coulomb_block(eri, (occ_a, vir_a), (occ_a, vir_a))
    bb = _coulomb_block(eri, (occ_b, vir_b), (occ_b, vir_b))
    ab = _coulomb_block(eri, (occ_a, vir_a), (occ_b, vir_b))

    hessian = np.zeros((sum(sizes), sum(sizes)))
    first = slice(0, sizes[0])
    second = slice(sizes[0], sum(sizes))
    if sizes[0]:
        ibja, ijab = _exchange_blocks(eri, occ_a, vir_a)
        hessian[first, first] = np.diag(_delta_epsilon(e_alpha, n_alpha)) + 2.0 * aa - ibja - ijab
    if sizes[1]:
        ibja, ijab = _exchange_blocks(eri, occ_b, vir_b)
        hessian[second, second] = np.diag(_delta_epsilon(e_beta, n_beta)) + 2.0 * bb - ibja - ijab
    if sizes[0] and sizes[1]:
        hessian[first, second] = 2.0 * ab
        hessian[second, first] = 2.0 * ab.T
    return StabilityResult(channels=(_channel("uhf->uhf", _lowest(hessian)),))


def _channel(name: str, lowest: float) -> StabilityChannel:
    return StabilityChannel(
        name=name, lowest_eigenvalue=lowest, stable=lowest >= -STABILITY_TOLERANCE
    )
