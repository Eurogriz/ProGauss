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
``ε_a − ε_i``.

Комплексные вращения
--------------------
Мнимые вращения ``C → C·exp(iY)`` с вещественной симметричной ``Y`` сохраняют
спиновую симметрию и ограничены на RHF→RHF и UHF→UHF. Энергия по ним растёт с
гессианом ``A − B`` (без Кулона: мнимая часть плотности не создаёт заряда):

.. math::

    (A−B)_{ia,jb} = δ_{ij}δ_{ab}(ε_a−ε_i) − (ij|ab) + (ib|ja)

Для DFT-гибрида обменные члены умножаются на долю точного обмена; XC-ядро в
``A − B`` не входит (мнимая часть ``D`` не меняет ``ρ``, ``∇ρ`` и ``τ`` в первом
порядке).

Общий путь: конечная разность орбитального градиента
-----------------------------------------------------
Аналитическое ``A + B`` выше выписано только для HF. Для DFT (нужно XC-ядро
второго порядка), ROHF (общие орбитали, три блока занятий) и комплексных
вращений ROHF есть общий механизм :func:`rotation_stability`: гессиан по
параметрам вращения ``C·exp(K)`` — это центральная конечная разность
**орбитального градиента** (он выражается через фокиан в смещённой точке),
поэтому ему нужен только построитель фокиана, а не ядро. Градиент по вращению
``(p, q)`` равен ``2 Σσ (nσ_p − nσ_q) Fσ_pq``; для мнимых — с мнимой частью
``F_pq``.
"""

from __future__ import annotations

from collections.abc import Callable
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
    imaginary = diagonal + ibja - ijab
    return StabilityResult(
        channels=(
            _channel("rhf->rhf", _lowest(singlet)),
            _channel("rhf->uhf", _lowest(triplet)),
            _channel("rhf->rhf:complex", _lowest(imaginary)),
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
    imaginary_lowest: list[float] = []
    if sizes[0]:
        ibja, ijab = _exchange_blocks(eri, occ_a, vir_a)
        delta = np.diag(_delta_epsilon(e_alpha, n_alpha))
        hessian[first, first] = delta + 2.0 * aa - ibja - ijab
        imaginary_lowest.append(_lowest(delta + ibja - ijab))
    if sizes[1]:
        ibja, ijab = _exchange_blocks(eri, occ_b, vir_b)
        delta = np.diag(_delta_epsilon(e_beta, n_beta))
        hessian[second, second] = delta + 2.0 * bb - ibja - ijab
        imaginary_lowest.append(_lowest(delta + ibja - ijab))
    if sizes[0] and sizes[1]:
        hessian[first, second] = 2.0 * ab
        hessian[second, first] = 2.0 * ab.T
    return StabilityResult(
        channels=(
            _channel("uhf->uhf", _lowest(hessian)),
            _channel("uhf->uhf:complex", min(imaginary_lowest)),
        )
    )


def _channel(name: str, lowest: float) -> StabilityChannel:
    return StabilityChannel(
        name=name, lowest_eigenvalue=lowest, stable=lowest >= -STABILITY_TOLERANCE
    )


# --------------------------------------------------------------------------- #
# Общий путь: конечная разность орбитального градиента
# --------------------------------------------------------------------------- #

#: Шаг конечной разности по параметру вращения, радиан. Ошибка центральной
#: разности ``O(h²)`` ≈ 1e-6 (относительно масштаба гессиана порядка 1), а
#: шум градиента 1e-12 даёт на таком шаге 5e-10 — оба много ниже порога 1e-4.
ROTATION_STEP = 1e-3

#: Строит спиновые фокианы по плотностям: ``(Dα, Dβ) → (Fα, Fβ, E)``.
SpinFock = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, float]]


@dataclass(frozen=True, slots=True)
class RotationSpace:
    """Пространство вращений: пары орбиталей и спиновые веса параметров.

    Параметр ``j`` вращает пару ``(p_j, q_j)`` канала α с весом ``weight_alpha_j``
    и канала β с весом ``weight_beta_j``. Так одним описанием покрываются
    синглетные (веса ``(1, 1)``), триплетные (``(1, −1)``), неограниченные
    (α-параметры ``(1, 0)``, β-параметры ``(0, 1)``) и ROHF-вращения (общая
    матрица ``C``, веса ``(1, 1)``; избыточные вращения внутри одинаково
    занятого блока дают нулевой градиент по определению ``nσ_p − nσ_q``).
    """

    p: np.ndarray
    q: np.ndarray
    weight_alpha: np.ndarray
    weight_beta: np.ndarray

    @property
    def size(self) -> int:
        """Число независимых параметров."""
        return int(self.p.size)


def restricted_space(n_occupied: int, n_orbitals: int, *, triplet: bool) -> RotationSpace:
    """Вращения «занятая → виртуальная» RHF/RKS (синглет — ``α=β``, триплет — ``α=−β``)."""
    pairs = [(i, a) for i in range(n_occupied) for a in range(n_occupied, n_orbitals)]
    count = len(pairs)
    return RotationSpace(
        p=np.array([pair[0] for pair in pairs], dtype=int),
        q=np.array([pair[1] for pair in pairs], dtype=int),
        weight_alpha=np.ones(count),
        weight_beta=-np.ones(count) if triplet else np.ones(count),
    )


def unrestricted_space(n_alpha: int, n_beta: int, n_orbitals: int) -> RotationSpace:
    """Независимые вращения каналов α и β (UHF, UKS)."""
    alpha = [(i, a) for i in range(n_alpha) for a in range(n_alpha, n_orbitals)]
    beta = [(i, a) for i in range(n_beta) for a in range(n_beta, n_orbitals)]
    pairs = alpha + beta
    return RotationSpace(
        p=np.array([pair[0] for pair in pairs], dtype=int),
        q=np.array([pair[1] for pair in pairs], dtype=int),
        weight_alpha=np.array([1.0] * len(alpha) + [0.0] * len(beta)),
        weight_beta=np.array([0.0] * len(alpha) + [1.0] * len(beta)),
    )


def rohf_space(n_alpha: int, n_beta: int, n_orbitals: int) -> RotationSpace:
    """Вращения между блоками ROHF (закрытый, открытый, виртуальный).

    Две орбитали входят в пространство, если их занятия ``(nα, nβ)`` различны;
    иначе вращение между ними не меняет ни ``Dα``, ни ``Dβ``.
    """
    alpha = np.arange(n_orbitals) < n_alpha
    beta = np.arange(n_orbitals) < n_beta
    pairs = [
        (p, q)
        for p in range(n_orbitals)
        for q in range(p + 1, n_orbitals)
        if (alpha[p], beta[p]) != (alpha[q], beta[q])
    ]
    count = len(pairs)
    return RotationSpace(
        p=np.array([pair[0] for pair in pairs], dtype=int),
        q=np.array([pair[1] for pair in pairs], dtype=int),
        weight_alpha=np.ones(count),
        weight_beta=np.ones(count),
    )


def _unitary(generator: np.ndarray, *, imaginary: bool) -> np.ndarray:
    """``exp(K)`` для вещественной антисимметричной ``K`` или ``exp(iY)`` для симметричной ``Y``.

    Оба случая — экспонента антиэрмитовой матрицы ``A = iH``; считается через
    собственное разложение эрмитовой ``H`` (точно унитарно, без ряда Тейлора).
    """
    # K = −iH для H = iK; A = iH для Y = H. Знак фазы различается.
    hermitian = generator if imaginary else 1j * generator
    values, vectors = np.linalg.eigh(hermitian)
    phase = np.exp(1j * values) if imaginary else np.exp(-1j * values)
    unitary = (vectors * phase) @ vectors.conj().T
    return np.asarray(unitary if imaginary else unitary.real)


def _rotation_gradient(
    fock: SpinFock,
    coefficients: tuple[np.ndarray, np.ndarray],
    occupations: tuple[np.ndarray, np.ndarray],
    space: RotationSpace,
    parameters: np.ndarray,
    *,
    imaginary: bool,
) -> np.ndarray:
    """Орбитальный градиент энергии в точке ``C·exp(K(x))``.

    ``dE/dx_j = Σσ w^σ_j Re Tr(G_j [nσ, Fσ_MO])``; для вещественного вращения
    ``G_pq = −G_qp = 1``, для мнимого ``G_pq = G_qp = i``.
    """
    n_orbitals = coefficients[0].shape[1]
    rotated: list[np.ndarray] = []
    for spin, weights in enumerate((space.weight_alpha, space.weight_beta)):
        generator = np.zeros((n_orbitals, n_orbitals))
        values = weights * parameters
        # Накопление, а не присваивание: пара ``(p, q)`` может встречаться у
        # нескольких параметров (α- и β-параметры UHF делят индексы пары).
        np.add.at(generator, (space.p, space.q), values)
        np.add.at(generator, (space.q, space.p), values if imaginary else -values)
        rotated.append(coefficients[spin] @ _unitary(generator, imaginary=imaginary))
    densities = [(c * n) @ c.conj().T for c, n in zip(rotated, occupations, strict=True)]
    focks = fock(densities[0], densities[1])[:2]

    gradient = np.zeros(space.size)
    for spin, weights in enumerate((space.weight_alpha, space.weight_beta)):
        c = rotated[spin]
        n = occupations[spin]
        f_mo = c.conj().T @ focks[spin] @ c
        commutator = n[:, None] * f_mo - f_mo * n[None, :]
        forward = commutator[space.q, space.p]
        backward = commutator[space.p, space.q]
        if imaginary:
            gradient += -weights * np.imag(forward + backward)
        else:
            gradient += weights * np.real(forward - backward)
    return gradient


def rotation_hessian(
    fock: SpinFock,
    coefficients: tuple[np.ndarray, np.ndarray],
    occupations: tuple[np.ndarray, np.ndarray],
    space: RotationSpace,
    *,
    imaginary: bool = False,
    step: float = ROTATION_STEP,
) -> np.ndarray:
    """Гессиан энергии по параметрам вращения: центральная разность градиента."""
    size = space.size
    hessian = np.zeros((size, size))
    for j in range(size):
        displacement = np.zeros(size)
        displacement[j] = step
        plus = _rotation_gradient(
            fock, coefficients, occupations, space, displacement, imaginary=imaginary
        )
        minus = _rotation_gradient(
            fock, coefficients, occupations, space, -displacement, imaginary=imaginary
        )
        hessian[:, j] = (plus - minus) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def rotation_stability(
    fock: SpinFock,
    alpha_coefficients: np.ndarray,
    beta_coefficients: np.ndarray,
    n_alpha: int,
    n_beta: int,
    *,
    kind: str,
    prefix: str,
    complex_rotations: bool = True,
) -> StabilityResult:
    """Устойчивость по конечной разности орбитального градиента.

    ``kind``:

    * ``restricted`` — RHF/RKS (``Cα = Cβ``, ``nα = nβ``): внутренний синглет
      ``x->x``, внешний триплет ``x->u`` и мнимый ``x->x:complex``;
    * ``unrestricted`` — UHF/UKS: ``u->u`` и ``u->u:complex``;
    * ``rohf`` — общие орбитали, разные занятия: ``rohf->rohf`` и ``…:complex``.

    Собственные значения нормированы как в аналитическом ``A ± B``: гессиан
    по параметру делится на число спин-каналов, которые параметр вращает (4 для
    RHF-синглета и ROHF — вращаются обе плотности по двум электронам; 2 для
    параметров UHF, где вращается одна плотность).
    """
    n_orbitals = alpha_coefficients.shape[1]
    occupations = (
        (np.arange(n_orbitals) < n_alpha).astype(float),
        (np.arange(n_orbitals) < n_beta).astype(float),
    )
    coefficients = (alpha_coefficients, beta_coefficients)

    plans: list[tuple[str, RotationSpace, float, bool]] = []
    if kind == "restricted":
        singlet = restricted_space(n_alpha, n_orbitals, triplet=False)
        plans = [
            (f"{prefix.lower()}->{prefix.lower()}", singlet, 4.0, False),
            (
                f"{prefix.lower()}->u{prefix[1:].lower()}",
                restricted_space(n_alpha, n_orbitals, triplet=True),
                4.0,
                False,
            ),
        ]
        if complex_rotations:
            plans.append((f"{prefix.lower()}->{prefix.lower()}:complex", singlet, 4.0, True))
    elif kind == "unrestricted":
        space = unrestricted_space(n_alpha, n_beta, n_orbitals)
        name = f"{prefix.lower()}->{prefix.lower()}"
        plans = [(name, space, 2.0, False)]
        if complex_rotations:
            plans.append((f"{name}:complex", space, 2.0, True))
    elif kind == "rohf":
        space = rohf_space(n_alpha, n_beta, n_orbitals)
        plans = [("rohf->rohf", space, 4.0, False)]
        if complex_rotations:
            plans.append(("rohf->rohf:complex", space, 4.0, True))
    else:
        msg = f"Неизвестный тип анализа устойчивости: {kind}"
        raise ValueError(msg)

    channels: list[StabilityChannel] = []
    for name, space, scale, imaginary in plans:
        if space.size == 0:
            continue
        hessian = rotation_hessian(fock, coefficients, occupations, space, imaginary=imaginary)
        channels.append(_channel(name, float(np.linalg.eigvalsh(hessian)[0]) / scale))
    return StabilityResult(channels=tuple(channels))
