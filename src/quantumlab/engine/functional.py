"""Обменно-корреляционные функционалы (§5 ТЗ).

Функционал — объект с декларативным описанием, а не ветка ``if`` внутри SCF:
это требование контракта ``ExchangeCorrelationFunctional`` и условие того, чтобы
добавление нового функционала не требовало правки решателя.

Здесь реализован LDA. GGA и гибриды строятся на том же интерфейсе: разница в
том, что ``evaluate`` получает ещё и градиент плотности, а гибрид возвращает
ненулевую долю точного обмена.

Все величины — в атомной системе единиц. ``exc`` — энергия на одну частицу,
``vxc`` — производная ``d(ρ ε_xc)/dρ``, то есть то, что входит в уравнения
Кона–Шэма.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, cartesian_powers
from quantumlab.engine.contracts import Array, XcEvaluation, XcEvaluationSpin
from quantumlab.engine.xc_spin_cores import (
    _lyp_spin_core,
    _pbe_spin_core,
    _vwn4rpa_spin_core,
    _vwn5_spin_core,
)
from quantumlab.errors import FunctionalNotFoundError

#: (3/π)^(1/3) — входит в обмен Слэтера.
_SLATER_FACTOR: float = (3.0 / np.pi) ** (1.0 / 3.0)

#: Коэффициент обмена Слэтера для **спиновой** LDA: ε_x = −C'_x(ρ_α^{4/3} +
#: ρ_β^{4/3})/ρ. Вдвое меньше «навыгляд», чем ¾(3/π)^(1/3), — и замена на
#: полный коэффициент занижает обмен ровно на 20.6 % (проверено: ошибка
#: выходит на 2.1e-01, а не на 1e-16).
_SPIN_SLATER_FACTOR: float = (3.0 / 8.0) * 2.0 ** (4.0 / 3.0) * (3.0 / np.pi) ** (1.0 / 3.0)

#: Знаменатель в s² обменной части PBE для спин-канала:
#: ``s_σ² = |∇ρ_σ|²/(D² ρ_σ^{8/3})``, ``D² = 24π^{4/3}/6^{1/3}``. Константа
#: восстановлена численно из эталонной реализации (fit по энергии, остаток
#: 7.9e-15 на сетке 126 точек), а не из текста: «школьное» определение
#: ``4(3π²)^{2/3}`` даёт 6.1 % ошибку, и форма F(s) при этом верна.
_PBE_SPIN_S2_DENOM: float = 24.0 * np.pi ** (4.0 / 3.0) / 6.0 ** (1.0 / 3.0)

#: Климпы переменных work-структуры — ровно те, что в C-коде libxc, с которым
#: сверены ядра (``xc_spin_cores``): при других клампах производные в
#: нулевых каналах дают бесконечные вклады, хотя энергия совпадает.
_SPIN_ZETA_EPS: float = 2.2204460492503131e-16
_SPIN_DENS_MIN: float = 1e-30
_SPIN_GRAD2_MIN: float = 1e-60

#: Порог плотности, ниже которого вклад считается нулевым. Без него деление на
#: ρ^(1/3) и логарифм r_s дают ``nan`` в хвостах плотности, и одна такая точка
#: отравляет всю матрицу Фока.
_DENSITY_FLOOR: float = 1e-14


def _spin_work(
    rho_alpha: np.ndarray,
    rho_beta: np.ndarray,
    s_aa: np.ndarray,
    s_ab: np.ndarray,
    s_bb: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Переменные work-структуры spin-GGA в установке libxc.

    Возвращает ``(dens, z, ds0, ds1, rs, sigmat, s00c, s11c, xt, xs0, xs1)`` —
    те же величины, что и в C-коде, из которого транскрибированы ядра
    :mod:`xc_spin_cores`, включая клампы. Цепное правило обязано использовать
    именно эти клампованные значения: подставь неклампованные — и в
    нулевых каналах появятся ложные вклады, невидимые по энергии.
    """
    dens = rho_alpha + rho_beta
    # В хвостовых точках сетки плотность может быть ровно 0 или ≤
    # _DENSITY_FLOOR: xt и rs считаются от клампованной плотности — как в
    # C-драйвере, где вход кламировался до вызова. Итоговые вклады там
    # маскируются в каждом функционале (``valid``), поэтому значение клампа
    # на результат не влияет — важно отсутствие inf/NaN в промежуточных.
    dens_safe = np.where(dens > _DENSITY_FLOOR, dens, 1.0)
    z = (rho_alpha - rho_beta) / dens_safe
    z = np.where(1.0 + z < _SPIN_ZETA_EPS, -1.0 + _SPIN_ZETA_EPS, z)
    z = np.where(1.0 - z < _SPIN_ZETA_EPS, 1.0 - _SPIN_ZETA_EPS, z)
    ds0 = np.maximum(rho_alpha, _SPIN_DENS_MIN)
    ds1 = np.maximum(rho_beta, _SPIN_DENS_MIN)
    rs = _wigner_seitz_radius(dens_safe)
    sigmat = np.maximum(s_aa + 2.0 * s_ab + s_bb, _SPIN_GRAD2_MIN)
    s00c = np.maximum(s_aa, _SPIN_GRAD2_MIN)
    s11c = np.maximum(s_bb, _SPIN_GRAD2_MIN)
    xt = np.sqrt(sigmat) / dens_safe ** (4.0 / 3.0)
    xs0 = np.sqrt(s00c) / ds0 ** (4.0 / 3.0)
    xs1 = np.sqrt(s11c) / ds1 ** (4.0 / 3.0)
    return dens, z, ds0, ds1, rs, sigmat, s00c, s11c, xt, xs0, xs1


def _spin_gga_chain(
    f: np.ndarray,
    dfdr: np.ndarray,
    dfdz: np.ndarray,
    dfdxt: np.ndarray,
    dfdxs0: np.ndarray,
    dfdxs1: np.ndarray,
    work: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Цепное правило spin-GGA: ``E_V = ρ·f(rs, z, xt, xs0, xs1)``.

    Возвращает производные энергии на единицу объёма
    ``(v_rho_a, v_rho_b, v_sigma_aa, v_sigma_ab, v_sigma_bb)``.
    Коэффициент 2 в ``v_sigma_ab`` уже учтён: ``s_tot`` зависит от
    ``s_ab = ∇ρ_α·∇ρ_β`` с множителем 2.
    """
    dens, z, ds0, ds1, rs, sigmat, s00c, s11c, xt, xs0, xs1 = work
    # Член xt идёт без множителя ρ: xt/∂ρ_σ = −4xt/(3ρ) и ρ·(−4xt/(3ρ)) = −4xt/3.
    # Лишний ρ здесь давал ошибку в v_ρ уровня 1e-2, невидимую по энергии.
    v_rho_a = (
        f
        - rs / 3.0 * dfdr
        + (1.0 - z) * dfdz
        - (4.0 / 3.0) * xt * dfdxt
        - (4.0 / 3.0) * (dens / ds0) * xs0 * dfdxs0
    )
    v_rho_b = (
        f
        - rs / 3.0 * dfdr
        - (1.0 + z) * dfdz
        - (4.0 / 3.0) * xt * dfdxt
        - (4.0 / 3.0) * (dens / ds1) * xs1 * dfdxs1
    )
    # Коэффициенты сверены с эталоном по всем членам: ∂xt/∂s_aa = xt/(2·s_tot),
    # ∂xt/∂s_ab = xt/s_tot (суммарный градиент зависит от s_ab с множителем 2),
    # ∂xs_σ/∂s_σσ = xs_σ/(2·s_σσ).
    v_xt = dens * xt / sigmat * dfdxt
    v_sigma_aa = 0.5 * v_xt + 0.5 * dens * xs0 / s00c * dfdxs0
    v_sigma_ab = v_xt
    v_sigma_bb = 0.5 * v_xt + 0.5 * dens * xs1 / s11c * dfdxs1
    return v_rho_a, v_rho_b, v_sigma_aa, v_sigma_ab, v_sigma_bb


class _SpinPart(Protocol):
    """Слагаемое функционала с ``evaluate_spin`` (для весовых сумм).

    Композитные функционалы суммируют части (обмен/корреляция) через
    ``_sum_spin_evaluations``; все части имеют один и тот же метод, поэтому
    типизация идёт протоколом, а не перечислением всех классов.
    """

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin: ...


def _sum_spin_evaluations(
    points: Array,
    density_spin: Array,
    density_gradient_spin: Array | None,
    terms: tuple[tuple[_SpinPart, float], ...],
) -> XcEvaluationSpin:
    """Взвешенная сумма спинных XC-слагаемых для композитных функционалов.

    ``terms`` — пары ``(функционал, вес)``. ``vsigma`` суммируется по тем
    слагаемым, которые его имеют (LDA-слагаемые дают ``None``); если его нет
    ни у одного — результат LDA и ``vsigma`` остаётся ``None``.
    """
    rho = np.asarray(density_spin, dtype=float)
    total = rho[0] + rho[1]
    valid = total > _DENSITY_FLOOR
    energy: np.ndarray | None = None
    vrho: np.ndarray | None = None
    vsigma: np.ndarray | None = None
    for functional, weight in terms:
        evaluation = functional.evaluate_spin(points, density_spin, density_gradient_spin)
        energy = (
            evaluation.energy_density * weight
            if energy is None
            else energy + weight * evaluation.energy_density
        )
        vrho = evaluation.vrho * weight if vrho is None else vrho + weight * evaluation.vrho
        if evaluation.vsigma is None:
            continue
        vsigma = (
            evaluation.vsigma * weight if vsigma is None else vsigma + weight * evaluation.vsigma
        )
    assert energy is not None and vrho is not None
    return XcEvaluationSpin(
        energy_density=np.where(valid, energy, 0.0),
        vrho=np.where(valid[None, :], vrho, 0.0),
        vsigma=None if vsigma is None else np.where(valid[None, None, :], vsigma, 0.0),
    )


def evaluate_basis(basis: BasisSet, molecule: Molecule, points: np.ndarray) -> np.ndarray:
    """Значения базисных функций в точках сетки.

    Возвращает массив ``(n_points, n_functions)``. Тонкая обёртка над
    :func:`evaluate_basis_with_gradients`: значения нужны и в LDA, где
    градиенты не используются вовсе.
    """
    values, _ = evaluate_basis_with_gradients(basis, molecule, points)
    return values


def evaluate_basis_with_gradients(
    basis: BasisSet, molecule: Molecule, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Значения базисных функций и их градиенты в точках сетки.

    Возвращает ``(values, gradients)`` форм ``(n_points, n_functions)`` и
    ``(n_points, n_functions, 3)``.

    Значения и градиенты считаются за один проход по оболочкам: радиальная
    часть ``exp(−ζr²)`` общая, и пересчитывать её для производных означало бы
    сделать самую дорогую часть работы дважды.

    Производная берётся по правилу произведения — дифференцируются и угловой
    многочлен, и экспонента:

    ``∂φ/∂x_a = (∂P/∂x_a)·C + P·(∂C/∂x_a)``,  ``∂C/∂x_a = −2 x_a Σ_i c_i ζ_i e^{−ζ_i r²}``

    Множители (коэффициенты сжатия, нормы примитивов, поправка на норму
    декартовой компоненты) те же, что при сборке интегралов, поэтому градиенты
    согласованы с самой функцией, а не с её «удобной» версией.

    Центры берутся из молекулы: ``BasisSet`` хранит только индекс атома в
    оболочке, сами координаты живут в структуре.
    """
    from quantumlab.engine.constants import angstrom_to_bohr

    centers = np.array(
        [[angstrom_to_bohr(value) for value in atom.position] for atom in molecule.atoms]
    )
    n_points = points.shape[0]
    columns: list[np.ndarray] = []
    gradient_columns: list[np.ndarray] = []
    for shell in basis.shells:
        center = centers[shell.center]
        delta = points - center[None, :]
        distance_squared = np.sum(delta * delta, axis=1)
        radial = np.zeros((shell.n_primitives, n_points))
        for index, exponent in enumerate(shell.exponents):
            radial[index] = np.exp(-exponent * distance_squared)
        contracted = shell.coefficients @ radial
        # ∂C/∂x_a = −2 x_a Σ_i c_i ζ_i exp(−ζ_i r²)
        radial_derivative = (np.asarray(shell.coefficients) * np.asarray(shell.exponents)) @ radial
        scales = shell.component_scales
        powers_list = list(cartesian_powers(shell.angular_momentum))
        for component, powers in enumerate(powers_list):
            angular = np.ones(n_points)
            for axis, power in enumerate(powers):
                if power:
                    angular *= delta[:, axis] ** power
            columns.append(angular * scales[component] * contracted)

            gradient = np.zeros((n_points, 3))
            for axis in range(3):
                # производная углового многочлена по этой оси
                if powers[axis]:
                    reduced = np.ones(n_points)
                    for other, power in enumerate(powers):
                        if other == axis:
                            if power > 1:
                                reduced *= delta[:, axis] ** (power - 1)
                            reduced *= power
                        elif power:
                            reduced *= delta[:, other] ** power
                    gradient[:, axis] = reduced * scales[component] * contracted
                # производная экспоненты
                gradient[:, axis] -= (
                    angular * scales[component] * 2.0 * delta[:, axis] * (radial_derivative)
                )
            gradient_columns.append(gradient)
    values = np.column_stack(columns) if columns else np.zeros((n_points, 0))
    gradients = (
        np.stack(gradient_columns, axis=1) if gradient_columns else np.zeros((n_points, 0, 3))
    )
    return values, gradients


def evaluate_basis_hessian_for_center(
    basis: BasisSet, molecule: Molecule, points: np.ndarray, center: int
) -> np.ndarray:
    """Гессианы базисных функций заданного центра, форма ``(n_points, n_bf_A, 3, 3)``.

    Нужны для обменно-корреляционного вклада в градиент GGA: производная
    ``σ = |∇ρ|²`` по координате ядра содержит ``∂_a∇φ_μ``. Возвращаются только
    функции выбранного центра и только он — полный тензор по всем функциям на
    ультрамелкой сетке занял бы сотни мегабайт, а в градиент он входит
    поатомно.

    Производная берётся дважды по правилу произведения:

    ``∂_b∂_a φ = (∂_b∂_a P)·C + (∂_a P)(∂_b C) + (∂_b P)(∂_a C) + P·(∂_b∂_a C)``

    где ``P`` — угловой многочлен, ``C`` — сжатая радиальная часть. Для
    экспоненты ``∂_b∂_a C = −2δ_ab Σ c_i ζ_i e^{−ζ_i r²} + 4Δ_aΔ_b Σ c_i ζ_i² e^{−ζ_i r²}``.
    """
    from quantumlab.engine.constants import angstrom_to_bohr

    centers = np.array(
        [[angstrom_to_bohr(value) for value in atom.position] for atom in molecule.atoms]
    )
    n_points = points.shape[0]
    columns: list[np.ndarray] = []
    for shell in basis.shells:
        if shell.center != center:
            continue
        offset = centers[shell.center]
        delta = points - offset[None, :]
        distance_squared = np.sum(delta * delta, axis=1)
        exponents = np.asarray(shell.exponents)
        coefficients = np.asarray(shell.coefficients)
        radial = np.zeros((shell.n_primitives, n_points))
        for index, exponent in enumerate(exponents):
            radial[index] = np.exp(-exponent * distance_squared)
        contracted = coefficients @ radial
        radial_first = (coefficients * exponents) @ radial
        radial_second = (coefficients * exponents**2) @ radial
        scales = shell.component_scales
        for component, powers in enumerate(cartesian_powers(shell.angular_momentum)):
            angular = np.ones(n_points)
            for axis, power in enumerate(powers):
                if power:
                    angular *= delta[:, axis] ** power

            # ∂_a P
            first_angular = []
            for axis in range(3):
                if powers[axis]:
                    reduced = np.full(n_points, float(powers[axis]))
                    for other, power in enumerate(powers):
                        if other == axis:
                            if power > 1:
                                reduced = reduced * delta[:, axis] ** (power - 1)
                        elif power:
                            reduced = reduced * delta[:, other] ** power
                    first_angular.append(reduced)
                else:
                    first_angular.append(np.zeros(n_points))

            hessian = np.zeros((n_points, 3, 3))
            for a in range(3):
                d_c_a = -2.0 * delta[:, a] * radial_first
                for b in range(3):
                    d_c_b = -2.0 * delta[:, b] * radial_first
                    # ∂_b∂_a P
                    if a == b:
                        if powers[a] >= 2:
                            second_angular = np.full(n_points, float(powers[a] * (powers[a] - 1)))
                            for other, power in enumerate(powers):
                                if other == a:
                                    second_angular = second_angular * delta[:, a] ** (power - 2)
                                elif power:
                                    second_angular = second_angular * delta[:, other] ** power
                        else:
                            second_angular = np.zeros(n_points)
                    elif powers[a] and powers[b]:
                        second_angular = np.full(n_points, float(powers[a] * powers[b]))
                        for other, power in enumerate(powers):
                            if other in (a, b):
                                if power > 1:
                                    second_angular = second_angular * delta[:, other] ** (power - 1)
                            elif power:
                                second_angular = second_angular * delta[:, other] ** power
                    else:
                        second_angular = np.zeros(n_points)

                    d_c_ab = (-2.0 * radial_first if a == b else 0.0) + 4.0 * delta[:, a] * delta[
                        :, b
                    ] * radial_second
                    hessian[:, a, b] = scales[component] * (
                        second_angular * contracted
                        + first_angular[a] * d_c_b
                        + first_angular[b] * d_c_a
                        + angular * d_c_ab
                    )
            columns.append(hessian)
    if not columns:
        return np.zeros((n_points, 0, 3, 3))
    return np.stack(columns, axis=1)


def density_at_points(values: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Плотность ``ρ(r) = Σ_μν D_μν φ_μ φ_ν`` в точках сетки.

    Симметрия матрицы плотности используется явно: диагональ входит один раз,
    недиагональные элементы — вдвое.
    """
    diagonal = np.einsum("pg,g,pg->p", values, np.diag(density), values)
    off_diagonal = np.einsum("pg,gh,ph->p", values, density - np.diag(np.diag(density)), values)
    return np.asarray(diagonal + off_diagonal)


def density_gradient_at_points(
    values: np.ndarray, gradients: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """Градиент плотности ``∇ρ(r)`` в точках сетки, форма ``(n_points, 3)``.

    ``∇ρ = Σ_μν D_μν (∇φ_μ φ_ν + φ_μ ∇φ_ν) = 2 Σ_μν D_μν ∇φ_μ φ_ν``

    Последнее равенство использует симметрию матрицы плотности; без него
    пришлось бы собирать вдвое больше слагаемых.
    """
    contracted = np.einsum("gh,ph->gp", density, values, optimize=True)
    return np.asarray(2.0 * np.einsum("pgd,gp->pd", gradients, contracted, optimize=True))


class LdaExchange:
    """Обмен Слэтера (ЛДА, Дирак).

    ``ε_x = −¾ (3/π)^(1/3) ρ^(1/3)``, ``v_x = −(3/π)^(1/3) ρ^(1/3)``.

    Точная формула для однородного электронного газа; в LDA она применяется
    локально. Никаких подгоночных параметров нет, поэтому она годится как
    проверка всей квадратурной обвязки независимо от корреляционной части.
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "slater"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "lda"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Энергия и потенциал обмена Слэтера в точках сетки."""
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        del points, density_gradient
        rho = np.asarray(density, dtype=float)
        safe = np.where(rho > _DENSITY_FLOOR, rho, 0.0)
        cube_root = np.cbrt(safe)
        exc = np.where(safe > 0.0, -0.75 * _SLATER_FACTOR * cube_root, 0.0)
        vrho = np.where(safe > 0.0, -_SLATER_FACTOR * cube_root, 0.0)
        return XcEvaluation(energy_density=np.asarray(exc), vrho=np.asarray(vrho))

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спин-обмен Слэтера: ``ε_x = −C'_x(ρ_α^{4/3} + ρ_β^{4/3})/ρ``.

        Сверка с эталоном (LDA_X, libxc 7.0.0) — в ``tests/test_engine_uks.py``,
        расхождение 2.1e-14. Коэффициент C'_x — :data:`_SPIN_SLATER_FACTOR`,
        а не ¾(3/π)^(1/3): подстановка полного коэффициента занижает обмен
        ровно на 20.6 %.
        """
        del points, density_gradient_spin
        rho = np.asarray(density_spin, dtype=float)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR
        per_volume = -_SPIN_SLATER_FACTOR * (
            np.where(rho[0] > _DENSITY_FLOOR, rho[0], 0.0) ** (4.0 / 3.0)
            + np.where(rho[1] > _DENSITY_FLOOR, rho[1], 0.0) ** (4.0 / 3.0)
        )
        energy_density = np.where(valid, per_volume / np.where(valid, total, 1.0), 0.0)
        vrho = np.zeros((2, total.shape[0]))
        for sigma in (0, 1):
            channel = np.where(rho[sigma] > _DENSITY_FLOOR, rho[sigma], 0.0)
            vrho[sigma] = np.where(
                rho[sigma] > _DENSITY_FLOOR,
                -(4.0 / 3.0) * _SPIN_SLATER_FACTOR * np.cbrt(channel),
                0.0,
            )
        return XcEvaluationSpin(energy_density=energy_density, vrho=vrho)


class VwnCorrelation:
    """Корреляция Воско–Вилка–Нусара (1980), неполяризованный случай.

    Интерполяция энергии однородного электронного газа по параметру
    ``r_s = (3/(4πρ))^(1/3)``; аргумент разложения — ``x = √r_s``.

    Константы соответствуют **параметризации V (RPA)** из таблицы статьи.
    В литературе под «VWN» встречаются два разных набора коэффициентов
    (собственная подгонка авторов и RPA-подгонка), и они дают энергии,
    различающиеся в четвёртом знаке. Какой именно набор использует
    сравниваемый код, определяется сверкой, а не по названию.
    """

    #: Коэффициенты RPA-параметризации для неполяризованного газа (VWN, 1980,
    #: таблица I). Значение ``A`` здесь — **полное**, равное 2 × 0.0310907:
    #: половина уже входит явным множителем ½ в формулу ``ε_c``, и подстановка
    #: 0.0310907 дала бы ровно вдвое заниженную корреляцию.
    A: float = 0.0621814
    X0: float = -0.10498
    B: float = 3.72744
    C: float = 12.9352

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "vwn"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "lda"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def _auxiliary(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """``X(x) = x² + bx + c`` и ``Q = √(4c − b²)``."""
        root = float(np.sqrt(4.0 * self.C - self.B**2))
        return x**2 + self.B * x + self.C, root

    @property
    def _denominator_at_x0(self) -> float:
        """``X(x₀)`` — значение знаменателя в точке вычета."""
        return self.X0**2 + self.B * self.X0 + self.C

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциал корреляционной части.

        Потенциал считается как ``v_c = ε_c − (r_s/3) dε_c/dr_s``, что
        эквивалентно ``d(ρ ε_c)/dρ`` через ``ρ ∝ r_s^{-3}``. Производная
        берётся аналитически, а не численно: конечная разность по ``r_s``
        в области малых плотностей дала бы шум, сравнимый с самой поправкой.
        """
        del tau

        del points, density_gradient, spin_polarized
        rho = np.asarray(density, dtype=float)
        valid = rho > _DENSITY_FLOOR
        r_s = np.zeros_like(rho)
        r_s[valid] = np.cbrt(3.0 / (4.0 * np.pi * rho[valid]))
        x = np.sqrt(r_s)

        denominator, root = self._auxiliary(x)
        denominator_x0 = self._denominator_at_x0
        arctan_term = np.arctan(root / (2.0 * x + self.B))
        slope = self.B * self.X0 / denominator_x0

        exc = np.zeros_like(rho)
        exc[valid] = (
            0.5
            * self.A
            * (
                np.log(x[valid] ** 2 / denominator[valid])
                + 2.0 * self.B / root * arctan_term[valid]
                - slope
                * (
                    np.log((x[valid] - self.X0) ** 2 / denominator[valid])
                    + 2.0 * (self.B + 2.0 * self.X0) / root * arctan_term[valid]
                )
            )
        )

        # dε/dx, затем v = ε − (r_s/3)·dε/dr_s = ε − (x/(6))·dε/dx, так как r_s = x².
        # Производная берётся только в валидных точках: при ρ = 0 величина x
        # равна нулю и член 2/x даёт деление на ноль.
        derivative = np.zeros_like(rho)
        derivative[valid] = self._dexc_dx(x[valid], denominator[valid], root)
        vrho = np.zeros_like(rho)
        vrho[valid] = exc[valid] - x[valid] / 6.0 * derivative[valid]
        return XcEvaluation(energy_density=exc, vrho=vrho)

    #: Спиновое ядро — транскрипция из libxc (maple2c/lda_c_vwn.c, func1);
    #: сверка с 7.0.0 — в docstring :mod:`xc_spin_cores` и в тестах UKS.
    #: Подкласс в RPA-параметризации заменяет его на своё.
    _spin_core = staticmethod(_vwn5_spin_core)

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спинная VWN-корреляция (LDA_C_VWN / LDA_C_VWN_RPA).

        Ядро — функция ``(rs, z)`` от полной плотности и ``z = (ρ_α−ρ_β)/ρ``;
        производные по каналам — цепным правилом от ``E_V = ρ·f``:

        ``v_ρ^σ = f − (r_s/3)·df/dr_s ± (1∓z)·df/dz``.

        Потенциалы сверены с эталоном до точности конечной разности (1e-13),
        энергия — до 1.2e-14; предельный случай ``z = 0`` обязан (и даёт)
        ровно неполяризованную VWN.
        """
        del points, density_gradient_spin
        rho = np.asarray(density_spin, dtype=float)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR
        z = (rho[0] - rho[1]) / np.where(valid, total, 1.0)
        z = np.clip(z, -1.0 + _SPIN_ZETA_EPS, 1.0 - _SPIN_ZETA_EPS)
        r_s = np.where(valid, total, 1.0)
        r_s = np.cbrt(3.0 / (4.0 * np.pi * r_s))
        f, dfdrs, dfdz = self._spin_core(r_s, z)
        base = f - r_s / 3.0 * dfdrs
        energy_density = np.where(valid, f, 0.0)
        vrho = np.zeros((2, total.shape[0]))
        vrho[0] = np.where(valid, base + (1.0 - z) * dfdz, 0.0)
        vrho[1] = np.where(valid, base - (1.0 + z) * dfdz, 0.0)
        return XcEvaluationSpin(energy_density=energy_density, vrho=vrho)

    def _dexc_dx(self, x: np.ndarray, denominator: np.ndarray, root: float) -> np.ndarray:
        """Аналитическая производная ``dε_c/dx``."""
        slope = self.B * self.X0 / self._denominator_at_x0
        log_part = 2.0 / x - (2.0 * x + self.B) / denominator
        arctan_derivative = -2.0 * root / (root**2 + (2.0 * x + self.B) ** 2)
        log_shifted = 2.0 / (x - self.X0) - (2.0 * x + self.B) / denominator
        return np.asarray(
            0.5
            * self.A
            * (
                log_part
                + 2.0 * self.B / root * arctan_derivative
                - slope * (log_shifted + 2.0 * (self.B + 2.0 * self.X0) / root * arctan_derivative)
            )
        )


class Svwn:
    """SVWN: обмен Слейтера плюс корреляция VWN-5.

    Суммирование двух реализаций протокола сделано отдельным классом, а не
    склейкой на месте вызова: иначе каждый решатель и каждый тест придумывал бы
    собственный способ собрать функционал, а они обязаны давать один и тот же
    результат.

    Это LDA-функционал: локальная плотность плотности, без градиентов.
    Спин-поляризованное вычисление идёт через ``evaluate_spin`` (ядро UKS);
    ``evaluate`` с ``spin_polarized=True`` отклоняется, чтобы не выдать
    непарную энергию по полной плотности.
    """

    name: str = "svwn"
    functional_class: str = "lda"
    is_hybrid: bool = False
    exact_exchange_fraction: float = 0.0

    def __init__(self) -> None:
        """Собирает обменную и корреляционную части."""
        self._exchange = LdaExchange()
        self._correlation = VwnCorrelation()

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Энергия и потенциал обмена+корреляции; см. протокол."""
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        exchange = self._exchange.evaluate(points, density, density_gradient)
        correlation = self._correlation.evaluate(points, density, density_gradient)
        return XcEvaluation(
            energy_density=exchange.energy_density + correlation.energy_density,
            vrho=exchange.vrho + correlation.vrho,
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Сумма спинного обмена Слэтера и спинной VWN-корреляции."""
        exchange = self._exchange.evaluate_spin(points, density_spin, density_gradient_spin)
        correlation = self._correlation.evaluate_spin(points, density_spin, density_gradient_spin)
        rho = np.asarray(density_spin, dtype=float)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR
        energy = exchange.energy_density + correlation.energy_density
        return XcEvaluationSpin(
            energy_density=np.where(valid, energy, 0.0),
            vrho=np.where(valid[None, :], exchange.vrho + correlation.vrho, 0.0),
        )


#: Параметры PW92 для неполяризованного газа (Perdew & Wang, PRB 45, 13244).
#: Одна и та же аналитическая форма годится для всех r_s: при r_s → 0 логарифм
#: даёт ``A ln r_s`` с правильным коэффициентом, при r_s → ∞ — требуемое
#: затухание. Две ветви — это PZ81, их с PW92 путать не следует.
_PW92_A = 0.031091
_PW92_ALPHA1 = 0.21370
_PW92_BETA1 = 7.5957
_PW92_BETA2 = 3.5876
_PW92_BETA3 = 1.6382
_PW92_BETA4 = 0.49294

#: Параметры PBE: κ и μ в обмене, γ и β в корреляции.
#: γ = (1 − ln 2)/π² выводится из RPA, а не подбирается.
_PBE_KAPPA = 0.804
#: Значение μ взято не из текста статьи PBE (там напечатано 0.21951), а из
#: эталонной реализации: оно восстановлено по данным LibXC и совпадает с ними
#: до 2.8e-17, тогда как печатное 0.21951 отличается на 5.0e-06 и даёт
#: расхождение в ε_x до 3.3e-07. Форма F(s) при этом проверена отдельно: κ и μ,
#: восстановленные из десяти разных пар точек, совпадают между собой, то есть
#: расхождение — в константе, а не в формуле.
_PBE_MU = 0.21951497276451748
_PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi**2
_PBE_BETA = 0.066725

#: Полноточный β из C-кода libxc (gga_c_pbe.c). Спиновое ядро сверялось с
#: эталоном именно на этом значении; округлённое 0.066725 выше дало бы
#: расхождение в ε_c порядка 1e-05 — невидимое по сходимости, видимое по
#: энергии.
_PBE_BETA_FULL = 0.06672455060314922


def _wigner_seitz_radius(rho: np.ndarray) -> np.ndarray:
    """Радиус Вигнера–Зейтца ``r_s = (3/(4πρ))^(1/3)``; вне валидных точек 0."""
    radius = np.zeros_like(rho)
    valid = rho > _DENSITY_FLOOR
    radius[valid] = np.cbrt(3.0 / (4.0 * np.pi * rho[valid]))
    return radius


class PwCorrelation:
    """Корреляция Пердью–Ванга 1992 (PW92), неполяризованный случай.

    ``ε_c = −2A(1 + α₁r_s) ln[1 + 1/(2A(β₁√r_s + β₂r_s + β₃r_s^1.5 + β₄r_s²))]``

    Именно на ней построен корреляционный член PBE, поэтому без неё «PBE» был бы
    другим функционалом: замена базовой LDA-корреляции на VWN меняет результат,
    и назвать его PBE значило бы выдать не то, что считаем (§54 ТЗ).
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "pw92"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "lda"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def energy_per_particle(self, r_s: np.ndarray) -> np.ndarray:
        """``ε_c(r_s)`` — нужно и самой корреляции, и PBE-надстройке."""
        polynomial = (
            _PW92_BETA1 * np.sqrt(r_s)
            + _PW92_BETA2 * r_s
            + _PW92_BETA3 * r_s**1.5
            + _PW92_BETA4 * r_s**2
        )
        return np.asarray(
            -2.0
            * _PW92_A
            * (1.0 + _PW92_ALPHA1 * r_s)
            * np.log1p(1.0 / (2.0 * _PW92_A * polynomial))
        )

    def _d_energy_d_r_s(self, r_s: np.ndarray) -> np.ndarray:
        """Аналитическая ``dε_c/dr_s``."""
        polynomial = (
            _PW92_BETA1 * np.sqrt(r_s)
            + _PW92_BETA2 * r_s
            + _PW92_BETA3 * r_s**1.5
            + _PW92_BETA4 * r_s**2
        )
        d_polynomial = (
            0.5 * _PW92_BETA1 / np.sqrt(r_s)
            + _PW92_BETA2
            + 1.5 * _PW92_BETA3 * np.sqrt(r_s)
            + 2.0 * _PW92_BETA4 * r_s
        )
        argument = 1.0 / (2.0 * _PW92_A * polynomial)
        # d/dr_s [ ln(1 + Q) ] = Q'/(1 + Q),  Q' = −Q·P'/P
        log_derivative = (-argument / (1.0 + argument)) * d_polynomial / polynomial
        return np.asarray(
            -2.0
            * _PW92_A
            * (_PW92_ALPHA1 * np.log1p(argument) + (1.0 + _PW92_ALPHA1 * r_s) * log_derivative)
        )

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциал корреляции PW92."""
        del tau

        del points, density_gradient, spin_polarized
        rho = np.asarray(density, dtype=float)
        valid = rho > _DENSITY_FLOOR
        r_s = _wigner_seitz_radius(rho)

        exc = np.zeros_like(rho)
        vrho = np.zeros_like(rho)
        exc[valid] = self.energy_per_particle(r_s[valid])
        # v = d(ρε)/dρ = ε − (r_s/3)·dε/dr_s, так как ρ ∝ r_s⁻³.
        vrho[valid] = exc[valid] - r_s[valid] / 3.0 * self._d_energy_d_r_s(r_s[valid])
        return XcEvaluation(energy_density=exc, vrho=vrho)


class PbeExchange:
    """Обмен PBE (GGA_X_PBE).

    ``ε_x = ε_x^LDA · F(s)``, ``F(s) = 1 + κ − κ/(1 + μs²/κ)``

    where ``s = |∇ρ|/(2 k_F ρ)`` — приведённый градиент плотности. При ``s → 0``
    ``F → 1`` и функционал переходит в LDA, при больших ``s`` растёт как ``s²``,
    что и есть градиентная поправка второго порядка.

    Потенциалы берутся по ``ρ`` и по ``σ = |∇ρ|²`` раздельно: именно так вариация
    по матрице плотности попадает в матрицу Фока.
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "pbe_x"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "gga"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциалы обмена PBE."""
        del tau

        del points, spin_polarized
        if density_gradient is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)

        rho = np.asarray(density, dtype=float)
        sigma = np.sum(np.asarray(density_gradient) ** 2, axis=1)
        valid = rho > _DENSITY_FLOOR

        # s² = σ/(4 k_F² ρ²), k_F = (3π²ρ)^(1/3)
        s_squared = np.zeros_like(rho)
        s_squared[valid] = sigma[valid] / (
            4.0 * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho[valid] ** (8.0 / 3.0)
        )
        denominator = 1.0 + _PBE_MU * s_squared / _PBE_KAPPA
        enhancement = 1.0 + _PBE_KAPPA - _PBE_KAPPA / denominator

        lda_density = -0.75 * _SLATER_FACTOR * np.cbrt(np.where(valid, rho, 0.0))
        lda_potential = -_SLATER_FACTOR * np.cbrt(np.where(valid, rho, 0.0))

        exc = np.where(valid, lda_density * enhancement, 0.0)
        # dF/ds² = μ/(1 + μs²/κ)²
        d_enhancement = np.where(valid, _PBE_MU / denominator**2, 0.0)

        vrho = np.where(
            valid,
            lda_potential * enhancement - lda_density * (8.0 / 3.0) * s_squared * d_enhancement,
            0.0,
        )
        # ∂s²/∂σ = s²/σ; при σ = 0 член равен нулю.
        vsigma = np.zeros_like(rho)
        nonzero = valid & (sigma > 0.0)
        vsigma[nonzero] = (
            rho[nonzero]
            * lda_density[nonzero]
            * d_enhancement[nonzero]
            * s_squared[nonzero]
            / sigma[nonzero]
        )
        return XcEvaluation(
            energy_density=np.asarray(exc), vrho=np.asarray(vrho), vsigma=np.asarray(vsigma)
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спинный обмен PBE: по каналу ``ε_x^σ = −C'_x ρ_σ^{4/3} F(s_σ)``.

        Определение ``s_σ`` отличается от неполяризованного (см.
        :data:`_PBE_SPIN_S2_DENOM`): «школьная» константа ``4(3π²)^{2/3}``
        даёт 6.1 % ошибку, эталонная ``24π^{4/3}/6^{1/3}`` — 7.9e-15.
        Сверка — в ``tests/test_engine_uks.py``.
        """
        del points
        if density_gradient_spin is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)
        rho = np.asarray(density_spin, dtype=float)
        grad = np.asarray(density_gradient_spin, dtype=float)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR

        per_volume = np.zeros_like(total)
        vrho = np.zeros((2, total.shape[0]))
        vsigma = np.zeros((2, 2, total.shape[0]))
        for sigma in (0, 1):
            active = rho[sigma] > _DENSITY_FLOOR
            channel = np.where(active, rho[sigma], 1.0)
            grad_norm2 = np.sum(grad[sigma] ** 2, axis=1)
            s_squared = np.zeros_like(total)
            s_squared[active] = grad_norm2[active] / (
                _PBE_SPIN_S2_DENOM * channel[active] ** (8.0 / 3.0)
            )
            denominator = 1.0 + _PBE_MU * s_squared / _PBE_KAPPA
            enhancement = 1.0 + _PBE_KAPPA - _PBE_KAPPA / denominator
            d_enhancement = _PBE_MU / denominator**2
            base = _SPIN_SLATER_FACTOR * channel ** (4.0 / 3.0)
            per_volume -= np.where(active, base * enhancement, 0.0)
            vrho[sigma] = np.where(
                active,
                -(4.0 / 3.0) * _SPIN_SLATER_FACTOR * np.cbrt(channel) * enhancement
                + (8.0 / 3.0) * _SPIN_SLATER_FACTOR * np.cbrt(channel) * s_squared * d_enhancement,
                0.0,
            )
            # ∂s²/∂σ_σσ = 6^{1/3}/(24π^{4/3} ρ_σ^{8/3}) = 1/(_PBE_SPIN_S2_DENOM·ρ_σ^{8/3})
            vsigma[sigma, sigma] = np.where(
                active,
                -_SPIN_SLATER_FACTOR * d_enhancement * channel ** (-4.0 / 3.0) / _PBE_SPIN_S2_DENOM,
                0.0,
            )
        return XcEvaluationSpin(
            energy_density=np.where(valid, per_volume / np.where(valid, total, 1.0), 0.0),
            vrho=vrho,
            vsigma=vsigma,
        )


class PbeCorrelation:
    """Корреляция PBE (GGA_C_PBE), неполяризованный случай.

    ``ε_c = ε_c^PW92 + H``, ``H = γ ln[1 + (β/γ)·(t² + A t⁴)/(1 + A t² + A² t⁴)]``

    ``A = (β/γ)/[exp(−ε_c^PW92/γ) − 1]``, ``t = |∇ρ|/(2 k_s ρ)``, ``k_s = √(4k_F/π)``.

    Для неполяризованного газа спиновый фактор ``φ = 1``. Производные по ``ρ``
    берутся с учётом того, что ``A`` тоже зависит от ``ρ`` — иначе потенциал
    расходится с энергией, и расхождение не видно ни по сходимости SCF, ни по
    коммутатору, только по энергии.
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "pbe_c"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "gga"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциалы корреляции PBE."""
        del tau

        del spin_polarized
        if density_gradient is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)

        rho = np.asarray(density, dtype=float)
        sigma = np.sum(np.asarray(density_gradient) ** 2, axis=1)
        valid = rho > _DENSITY_FLOOR

        lda = PwCorrelation().evaluate(points, density)
        lda_exc = lda.energy_density
        lda_vrho = lda.vrho

        # t² = σ π/(16 k_F ρ²), k_F = (3π²ρ)^(1/3); k_s² = 4k_F/π, φ = 1.
        t_squared = np.zeros_like(rho)
        t_squared[valid] = (
            np.pi
            * sigma[valid]
            / (16.0 * (3.0 * np.pi**2) ** (1.0 / 3.0) * rho[valid] ** (7.0 / 3.0))
        )

        exponent = np.zeros_like(rho)
        exponent[valid] = np.exp(-lda_exc[valid] / _PBE_GAMMA) - 1.0
        coefficient = np.zeros_like(rho)
        coefficient[valid] = (_PBE_BETA / _PBE_GAMMA) / exponent[valid]

        numerator = t_squared + coefficient * t_squared**2
        denominator = 1.0 + coefficient * t_squared + coefficient**2 * t_squared**2
        fraction = np.where(valid, numerator / denominator, 0.0)
        h_term = np.where(valid, _PBE_GAMMA * np.log1p((_PBE_BETA / _PBE_GAMMA) * fraction), 0.0)

        exc = lda_exc + h_term

        # ∂H/∂t² = β(1 + 2Au)/(1 + Au + A²u²)² / (1 + (β/γ)·fraction),  u = t²
        common = 1.0 + (_PBE_BETA / _PBE_GAMMA) * fraction
        d_fraction_d_t = np.where(
            valid, (1.0 + 2.0 * coefficient * t_squared) / denominator**2, 0.0
        )
        d_h_d_t = _PBE_BETA * d_fraction_d_t / common
        # ∂D/∂A = −A u³(2 + Au)/(1 + Au + A²u²)²,  u = t².
        # Здесь легко потерять степень u: энергия этой производной не пользуется
        # вовсе, поэтому ошибка видна только в потенциале и только по энергии.
        d_fraction_d_a = np.where(
            valid,
            -coefficient * t_squared**3 * (2.0 + coefficient * t_squared) / denominator**2,
            0.0,
        )
        d_h_d_a = _PBE_BETA * d_fraction_d_a / common

        # dA/dρ через dε_c^LDA/dρ = (v_c^LDA − ε_c^LDA)/ρ
        d_coefficient_d_rho = np.zeros_like(rho)
        d_coefficient_d_rho[valid] = (
            (_PBE_BETA / _PBE_GAMMA**2)
            * (exponent[valid] + 1.0)
            / exponent[valid] ** 2
            * (lda_vrho[valid] - lda_exc[valid])
            / rho[valid]
        )

        vrho = np.where(
            valid,
            lda_vrho
            + h_term
            + rho
            * (
                d_h_d_t * (-7.0 / 3.0 * t_squared / np.where(valid, rho, 1.0))
                + d_h_d_a * d_coefficient_d_rho
            ),
            0.0,
        )

        vsigma = np.zeros_like(rho)
        nonzero = valid & (sigma > 0.0)
        vsigma[nonzero] = rho[nonzero] * d_h_d_t[nonzero] * t_squared[nonzero] / sigma[nonzero]
        return XcEvaluation(
            energy_density=np.asarray(exc), vrho=np.asarray(vrho), vsigma=np.asarray(vsigma)
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спинная корреляция PBE (GGA_C_PBE, поляризованный случай).

        Ядро — транскрипция maple2c-кода libxc (сверка с 7.0.0 — до машинной
        точности, см. :mod:`xc_spin_cores`). Зависит только от ``s_tot =
        |∇ρ_tot|²`` и собственных ``s_σσ`` каналов — межканальных членов
        ``∇ρ_α·∇ρ_β`` у функционала нет, но они входят в ``s_tot``, поэтому
        ``vsigma[αβ]`` ненулевой. Параметры — полноточные из C-кода:
        ``β = 0.06672455060314922`` (а не округлённое 0.066725 из
        неполяризованного пути).
        """
        del points
        if density_gradient_spin is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)
        rho = np.asarray(density_spin, dtype=float)
        grad = np.asarray(density_gradient_spin, dtype=float)
        s_aa = np.sum(grad[0] ** 2, axis=1)
        s_ab = np.sum(grad[0] * grad[1], axis=1)
        s_bb = np.sum(grad[1] ** 2, axis=1)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR

        work = _spin_work(rho[0], rho[1], s_aa, s_ab, s_bb)
        dens, z, _ds0, _ds1, rs, _sigmat, _s00c, _s11c, xt, xs0, xs1 = work
        # Ядро — генерируемая транскрипция без аннотаций (см. :mod:`xc_spin_cores`).
        f, dfdrs, dfdz, dfdxt, dfdxs0, dfdxs1 = _pbe_spin_core(  # type: ignore[no-untyped-call]
            _PBE_BETA_FULL, _PBE_GAMMA, 1.0, dens, z, rs, xt, xs0, xs1
        )
        v_rho_a, v_rho_b, vs_aa, vs_ab, vs_bb = _spin_gga_chain(
            f, dfdrs, dfdz, dfdxt, dfdxs0, dfdxs1, work
        )
        vrho = np.stack([v_rho_a, v_rho_b], axis=0)
        vsigma = np.stack(
            [np.stack([vs_aa, vs_ab], axis=0), np.stack([vs_ab, vs_bb], axis=0)], axis=0
        )
        return XcEvaluationSpin(
            energy_density=np.where(valid, f, 0.0),
            vrho=np.where(valid[None, :], vrho, 0.0),
            vsigma=np.where(valid[None, None, :], vsigma, 0.0),
        )


class Pbe:
    """PBE: обмен PBE + корреляция PBE (на базе PW92).

    Соответствует комбинации LibXC ``LDA_X + LDA_C_PZ + GGA_X_PBE + GGA_C_PBE``
    с той оговоркой, что обменная LDA-часть у нас Слейтера, а корреляционная
    базовая — PW92, как и в ``GGA_C_PBE``.
    """

    name: str = "pbe"
    functional_class: str = "gga"
    is_hybrid: bool = False
    exact_exchange_fraction: float = 0.0
    requires_tau: bool = False

    def __init__(self) -> None:
        """Собирает обменную и корреляционную части."""
        self._exchange = PbeExchange()
        self._correlation = PbeCorrelation()

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Энергия и потенциалы PBE; см. протокол."""
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        exchange = self._exchange.evaluate(points, density, density_gradient)
        correlation = self._correlation.evaluate(points, density, density_gradient)
        vsigma = None
        if exchange.vsigma is not None and correlation.vsigma is not None:
            vsigma = exchange.vsigma + correlation.vsigma
        return XcEvaluation(
            energy_density=exchange.energy_density + correlation.energy_density,
            vrho=exchange.vrho + correlation.vrho,
            vsigma=vsigma,
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Сумма спинного обмена PBE и спинной корреляции PBE."""
        return _sum_spin_evaluations(
            points,
            density_spin,
            density_gradient_spin,
            ((self._exchange, 1.0), (self._correlation, 1.0)),
        )


class Pbe0:
    """PBE0 (PBE1PBE): ¼ точного обмена + ¾ обмена PBE + корреляция PBE.

    ``E_xc = ¼ E_x^exact + ¾ E_x^PBE + E_c^PBE``

    Доля точного обмена не подгонялась: она следует из требования, чтобы
    функционал четвёртого порядка градиентного разложения совпадал с
    известным результатом теории возмущений. Это делает PBE0 «беспараметрическим»
    гибридом — в отличие от B3LYP, где три коэффициента получены фитом.

    Точный обмен здесь не считается: ``evaluate`` возвращает только DFT-часть,
    а долю ``α`` решатель подставляет сам, собирая ``F = H + J + V_xc − αK`` и
    добавляя в энергию ``−¼α·D:K``. Разделение такое потому, что K не зависит
    от квадратурной сетки, и считать его внутри функционала означало бы
    дублировать работу решателя.
    """

    name: str = "pbe0"
    functional_class: str = "hybrid"
    is_hybrid: bool = True
    exact_exchange_fraction: float = 0.25
    requires_tau: bool = False

    #: Доля полунелокального обмена PBE: дополняет точный обмен до единицы.
    dft_exchange_fraction: float = 0.75

    def __init__(self) -> None:
        """Собирает обменную и корреляционную части."""
        self._exchange = PbeExchange()
        self._correlation = PbeCorrelation()

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """DFT-часть PBE0; см. протокол.

        Обмен умножается на ¾ — именно эта часть сочетается с ¼ точного обмена.
        Корреляция входит целиком: в PBE0 её не масштабируют.
        """
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        exchange = self._exchange.evaluate(points, density, density_gradient)
        correlation = self._correlation.evaluate(points, density, density_gradient)
        weight = self.dft_exchange_fraction
        vsigma = None
        if exchange.vsigma is not None:
            vsigma = weight * exchange.vsigma + (
                correlation.vsigma if correlation.vsigma is not None else 0.0
            )
        return XcEvaluation(
            energy_density=weight * exchange.energy_density + correlation.energy_density,
            vrho=weight * exchange.vrho + correlation.vrho,
            vsigma=vsigma,
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """DFT-часть PBE0 для UKS: ¾ обмена PBE + корреляция PBE целиком."""
        return _sum_spin_evaluations(
            points,
            density_spin,
            density_gradient_spin,
            ((self._exchange, self.dft_exchange_fraction), (self._correlation, 1.0)),
        )


# --------------------------------------------------------------------------- #
# B88 и LYP: ингредиенты BLYP и B3LYP
# --------------------------------------------------------------------------- #

#: Параметр Беке (1988). Единственный параметр B88; подобран на атомах.
_B88_BETA = 0.0042

#: Константы Colle–Salvetti в записи LYP (Lee, Yang, Parr 1988). В литературе
#: ``a`` и ``b`` встречаются и в обратном порядке, поэтому значения сверены с
#: LibXC численно, а не переписаны по названию: форма с ``a = 0.04918`` перед
#: первым членом воспроизводит ``GGA_C_LYP`` до 2.8e-16.
_LYP_A = 0.04918
_LYP_B = 0.132
_LYP_C = 0.2533
_LYP_D = 0.349

#: ``C_F = (3/10)(3π²)^(2/3)`` — коэффициент кинетической энергии Томаса–Ферми.
_LYP_CF = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)


class BeckeExchange:
    """Обмен B88: LDA плюс градиентная поправка Беке (1988).

    ``ε_x^σ = ε_x^{LDA}(ρ_σ) − β ρ_σ^{4/3} x_σ²/(1 + 6β x_σ asinh x_σ)``,
    ``x_σ = |∇ρ_σ|/ρ_σ^{4/3}``, ``β = 0.0042``.

    Поправка считается по спиновым каналам: ``ρ_σ = ρ/2``, ``|∇ρ_σ| = |∇ρ|/2``,
    результат суммируется по двум каналам. Это не формальность: подстановка
    полной плотности вместо ``ρ/2`` расходится с LibXC на 1.5e-02, тогда как
    спиновая запись совпадает до 5.6e-16.

    Класс возвращает обмен **целиком**, включая LDA, — так же устроен
    ``GGA_X_B88`` в LibXC. Именно поэтому B3LYP собирается как
    ``0.08·LDA + 0.72·B88``: слагаемые LDA дают ``0.08 + 0.72 = 0.80``, что и
    есть ``(1 − a₀)`` при ``a₀ = 0.20``, а от B88 остаётся только градиентная
    поправка с весом ``0.72``.
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "b88_x"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "gga"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциалы обмена B88."""
        del tau

        del points, spin_polarized
        if density_gradient is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)

        rho = np.asarray(density, dtype=float)
        sigma = np.sum(np.asarray(density_gradient) ** 2, axis=1)
        valid = rho > _DENSITY_FLOOR

        safe_rho = np.where(valid, rho, 1.0)
        safe_sigma = np.where(valid, sigma, 0.0)

        # Спиновый канал: rho_s = rho/2, |grad rho_s| = |grad rho|/2.
        channel = 0.5 * safe_rho
        gradient = 0.5 * np.sqrt(safe_sigma)
        x = np.zeros_like(rho)
        x[valid] = gradient[valid] / channel[valid] ** (4.0 / 3.0)

        denominator = 1.0 + 6.0 * _B88_BETA * x * np.arcsinh(x)
        # F(x) = x²/(1 + 6βx·asinh x) и её производная по x.
        correction = x**2 / denominator
        d_correction = (
            2.0 * x * denominator
            - x**2 * 6.0 * _B88_BETA * (np.arcsinh(x) + x / np.sqrt(1.0 + x**2))
        ) / denominator**2

        # k — энергия на единицу объёма. Слагаемые собраны по-разному, и это не
        # произвол: LDA берётся по полной плотности, а градиентная поправка — по
        # спиновым каналам. Именно так устроен GGA_X_B88 в LibXC; попытка удвоить
        # и LDA по каналам даёт расхождение 2.4e-01.
        base = channel ** (4.0 / 3.0)
        k = -0.75 * _SLATER_FACTOR * safe_rho ** (4.0 / 3.0) - 2.0 * _B88_BETA * base * correction

        # Производные по rho и sigma через цепочку x -> (rho_s, |grad rho_s|).
        dx_drho = np.zeros_like(rho)
        dx_drho[valid] = -(4.0 / 3.0) * gradient[valid] / channel[valid] ** (7.0 / 3.0) * 0.5
        dx_dsigma = np.zeros_like(rho)
        nonzero = valid & (safe_sigma > 0.0)
        dx_dsigma[nonzero] = (
            1.0 / channel[nonzero] ** (4.0 / 3.0) / (4.0 * np.sqrt(safe_sigma[nonzero]))
        )

        vrho = -_SLATER_FACTOR * safe_rho ** (1.0 / 3.0) - 2.0 * _B88_BETA * (
            (4.0 / 3.0) * channel ** (1.0 / 3.0) * 0.5 * correction + base * d_correction * dx_drho
        )
        vsigma = -2.0 * _B88_BETA * base * d_correction * dx_dsigma

        energy = np.zeros_like(rho)
        energy[valid] = k[valid] / rho[valid]
        return XcEvaluation(
            energy_density=energy,
            vrho=np.where(valid, vrho, 0.0),
            vsigma=np.where(valid, vsigma, 0.0),
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спинный обмен B88, формула по каналу.

        ``ε_x^σ = −C'_x ρ_σ^{4/3} − β ρ_σ^{4/3} x_σ²/(1 + 6β x_σ asinh x_σ)``.

        Форма сверена с эталоном на 84 точках (включая антипараллельные
        градиенты) до 6.4e-16. Производные по ``s_σσ`` записаны через
        ``H'(x)/x``, чтобы не иметь деления на ``√s_σσ``: при ``∇ρ_σ = 0``
        предел конечен и равен ``−(β/2)·2/ρ_σ^{4/3}``, и именно он
        воспроизводится, а не шум от клампа.
        """
        del points
        if density_gradient_spin is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)
        rho = np.asarray(density_spin, dtype=float)
        grad = np.asarray(density_gradient_spin, dtype=float)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR

        per_volume = np.zeros_like(total)
        vrho = np.zeros((2, total.shape[0]))
        vsigma = np.zeros((2, 2, total.shape[0]))
        for sigma in (0, 1):
            active = rho[sigma] > _DENSITY_FLOOR
            channel = np.where(active, rho[sigma], 1.0)
            grad_norm2 = np.maximum(np.sum(grad[sigma] ** 2, axis=1), _SPIN_GRAD2_MIN)
            x = np.sqrt(grad_norm2) / channel ** (4.0 / 3.0)
            asinh_x = np.arcsinh(x)
            denom = 1.0 + 6.0 * _B88_BETA * x * asinh_x
            correction = x * x / denom
            # H'(x)/x — стабильна при x → 0 (предел 2), делений на x нет.
            dcorr_over_x = (
                2.0 * denom - x * 6.0 * _B88_BETA * (asinh_x + x / np.sqrt(1.0 + x * x))
            ) / denom**2
            base = channel ** (4.0 / 3.0)
            b88_term = _SPIN_SLATER_FACTOR * base + _B88_BETA * base * correction
            per_volume -= np.where(active, b88_term, 0.0)
            vrho[sigma] = np.where(
                active,
                -(4.0 / 3.0) * _SPIN_SLATER_FACTOR * np.cbrt(channel)
                - (4.0 / 3.0) * _B88_BETA * np.cbrt(channel) * (correction - x * x * dcorr_over_x),
                0.0,
            )
            vsigma[sigma, sigma] = np.where(active, -0.5 * _B88_BETA * dcorr_over_x / base, 0.0)
        return XcEvaluationSpin(
            energy_density=np.where(valid, per_volume / np.where(valid, total, 1.0), 0.0),
            vrho=vrho,
            vsigma=vsigma,
        )


class LypCorrelation:
    """Корреляция LYP (Colle–Salvetti в форме Lee–Yang–Parr, 1988).

    Запись через полную плотность и ``σ = |∇ρ|²`` — та, что воспроизводит
    ``GGA_C_LYP`` из LibXC::

        k = −a ρ Z − a b ω ρ² [ C_F ρ^{8/3} − σ (1/24 + 7δ/72) ]
        Z = (1 + d ρ^{−1/3})^{−1},  δ = (c + d Z) ρ^{−1/3}
        ω = e^{−c ρ^{−1/3}} Z ρ^{−11/3}

    Локальная часть (``σ = 0``) совпадает с эталоном до 2.8e-16. Коэффициент
    при ``σ`` и определение ``δ`` восстановлены по эталону и проверены на сетке
    из 40 плотностей от 0.05 до 20: разложение по базису ``{1, δ, δ²}`` даёт
    1/24 и 7/72 при остатке 3.8e-11, то есть форма определена однозначно.
    ``δ`` здесь — величина из спин-поляризованной записи Molpro,
    ``(c + dZ) ρ^{−1/3}``, а не ``c ρ^{−1/3} Z`` из замкнутой: при ``σ = 0`` обе
    дают одинаковую энергию, поэтому локальная часть выбрать между ними не
    может, и подстановка «замкнутого» ``δ`` расходится на десятки процентов.

    Здесь ``k`` — энергия на единицу объёма, то есть ``ρ ε_c``. Лапласиан
    плотности, присутствующий в исходной формуле Colle–Salvetti, исключён
    интегрированием по частям: поэтому функционал остаётся GGA и не требует
    вторых производных плотности на сетке.
    """

    requires_tau: bool = False

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "lyp_c"

    @property
    def functional_class(self) -> str:
        """Класс функционала."""
        return "gga"

    @property
    def is_hybrid(self) -> bool:
        """Точного обмена нет."""
        return False

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена."""
        return 0.0

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
        tau: Array | None = None,
    ) -> XcEvaluation:
        """Энергия и потенциалы корреляции LYP."""
        del tau

        del points, spin_polarized
        if density_gradient is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)

        rho = np.asarray(density, dtype=float)
        sigma = np.sum(np.asarray(density_gradient) ** 2, axis=1)
        valid = rho > _DENSITY_FLOOR

        r = np.where(valid, rho, 1.0)
        s = np.where(valid, sigma, 0.0)

        cube_root = r ** (1.0 / 3.0)
        inverse = 1.0 / cube_root
        z_factor = 1.0 / (1.0 + _LYP_D * inverse)
        delta = (_LYP_C + _LYP_D * z_factor) * inverse
        omega = np.exp(-_LYP_C * inverse) * z_factor * r ** (-11.0 / 3.0)
        bracket = 1.0 / 24.0 + 7.0 * delta / 72.0

        k = (
            -_LYP_A * r * z_factor
            - _LYP_A * _LYP_B * _LYP_CF * omega * r ** (14.0 / 3.0)
            + _LYP_A * _LYP_B * omega * r**2 * s * bracket
        )

        # Производные по rho. Обозначения совпадают с формулами выше.
        d_z = _LYP_D * z_factor**2 / (3.0 * r ** (4.0 / 3.0))
        # dδ/dρ для δ = (c + dZ) ρ^{−1/3}; dZ/dρ = d Z²/(3 ρ^{4/3}).
        d_delta = (_LYP_D**2 * z_factor**2 * inverse - (_LYP_C + _LYP_D * z_factor)) / (
            3.0 * r ** (4.0 / 3.0)
        )
        d_bracket = (7.0 / 72.0) * d_delta
        d_omega = omega * (
            (_LYP_C + _LYP_D * z_factor) / (3.0 * r ** (4.0 / 3.0)) - (11.0 / 3.0) / r
        )

        product = _LYP_A * _LYP_B
        vrho = (
            -_LYP_A * (z_factor + r * d_z)
            - product
            * _LYP_CF
            * (d_omega * r ** (14.0 / 3.0) + omega * (14.0 / 3.0) * r ** (11.0 / 3.0))
            + product
            * s
            * (d_omega * r**2 * bracket + omega * 2.0 * r * bracket + omega * r**2 * d_bracket)
        )
        vsigma = product * omega * r**2 * bracket

        energy = np.zeros_like(rho)
        energy[valid] = k[valid] / rho[valid]
        return XcEvaluation(
            energy_density=energy,
            vrho=np.where(valid, vrho, 0.0),
            vsigma=np.where(valid, vsigma, 0.0),
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спинная корреляция LYP (GGA_C_LYP, поляризованный случай).

        Ядро — транскрипция maple2c-кода libxc (сверка с 7.0.0 — до машинной
        точности, см. :mod:`xc_spin_cores`); цепное правило — общее для
        спин-GGA (:func:`_spin_gga_chain`).
        """
        del points
        if density_gradient_spin is None:
            msg = "GGA-функционал требует градиент плотности; передать None нельзя."
            raise ValueError(msg)
        rho = np.asarray(density_spin, dtype=float)
        grad = np.asarray(density_gradient_spin, dtype=float)
        s_aa = np.sum(grad[0] ** 2, axis=1)
        s_ab = np.sum(grad[0] * grad[1], axis=1)
        s_bb = np.sum(grad[1] ** 2, axis=1)
        total = rho[0] + rho[1]
        valid = total > _DENSITY_FLOOR

        work = _spin_work(rho[0], rho[1], s_aa, s_ab, s_bb)
        dens, z, _ds0, _ds1, rs, _sigmat, _s00c, _s11c, xt, xs0, xs1 = work
        # Ядро — генерируемая транскрипция без аннотаций (см. :mod:`xc_spin_cores`).
        f, dfdrs, dfdz, dfdxt, dfdxs0, dfdxs1 = _lyp_spin_core(  # type: ignore[no-untyped-call]
            _LYP_A, _LYP_B, _LYP_C, _LYP_D, dens, z, rs, xt, xs0, xs1
        )
        v_rho_a, v_rho_b, vs_aa, vs_ab, vs_bb = _spin_gga_chain(
            f, dfdrs, dfdz, dfdxt, dfdxs0, dfdxs1, work
        )
        vrho = np.stack([v_rho_a, v_rho_b], axis=0)
        vsigma = np.stack(
            [np.stack([vs_aa, vs_ab], axis=0), np.stack([vs_ab, vs_bb], axis=0)], axis=0
        )
        return XcEvaluationSpin(
            energy_density=np.where(valid, f, 0.0),
            vrho=np.where(valid[None, :], vrho, 0.0),
            vsigma=np.where(valid[None, None, :], vsigma, 0.0),
        )


class VwnRpaCorrelation(VwnCorrelation):
    """Корреляция VWN в параметризации IV (RPA).

    Отличается от :class:`VwnCorrelation` только тремя коэффициентами
    разложения. Это не косметика: B3LYP по построению смешивает LYP именно с
    RPA-параметризацией, и подстановка VWN5 меняет энергию на 3.8e-03 э —
    заметно больше, чем расхождение с эталоном у всех прочих функционалов.
    """

    X0: float = -0.409286
    B: float = 13.0720
    C: float = 42.7198

    #: Спиновое ядро RPA-параметризации (maple2c/lda_c_vwn_rpa.c, func1).
    _spin_core = staticmethod(_vwn4rpa_spin_core)

    @property
    def name(self) -> str:
        """Имя функционала."""
        return "vwn_rpa"


class Blyp:
    """BLYP: обмен B88 плюс корреляция LYP."""

    name: str = "blyp"
    functional_class: str = "gga"
    is_hybrid: bool = False
    exact_exchange_fraction: float = 0.0
    requires_tau: bool = False

    def __init__(self) -> None:
        """Собирает обменную и корреляционную части."""
        self._exchange = BeckeExchange()
        self._correlation = LypCorrelation()

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Сумма обмена B88 и корреляции LYP."""
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        exchange = self._exchange.evaluate(points, density, density_gradient)
        correlation = self._correlation.evaluate(points, density, density_gradient)
        vsigma = None
        if exchange.vsigma is not None:
            vsigma = exchange.vsigma + (
                correlation.vsigma if correlation.vsigma is not None else 0.0
            )
        return XcEvaluation(
            energy_density=exchange.energy_density + correlation.energy_density,
            vrho=exchange.vrho + correlation.vrho,
            vsigma=vsigma,
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Сумма спинного обмена B88 и спинной корреляции LYP."""
        return _sum_spin_evaluations(
            points,
            density_spin,
            density_gradient_spin,
            ((self._exchange, 1.0), (self._correlation, 1.0)),
        )


class B3lyp:
    """B3LYP (Becke 1993, три параметра).

    ``E_xc = 0.80·E_x^LSDA + 0.20·E_x^HF + 0.72·ΔE_x^B88 + 0.81·E_c^LYP + 0.19·E_c^VWN(RPA)``

    В коде обмен собран иначе, но тождественно: ``B88`` здесь возвращается
    вместе со своим LDA-слагаемым, поэтому ``0.08·LDA + 0.72·B88`` раскрывается
    в ``0.80·LDA + 0.72·ΔB88``.

    Две детали, которые легко упустить. Во-первых, LYP смешивается с VWN через
    коэффициент 0.81, а не добавляется к уже имеющейся корреляции: LYP сам
    содержит локальную часть, и двойной учёт завысил бы корреляцию. Во-вторых,
    VWN обязан быть в RPA-параметризации — с VWN5 энергия уходит на 3.8e-03 э.
    """

    name: str = "b3lyp"
    functional_class: str = "hybrid"
    is_hybrid: bool = True
    exact_exchange_fraction: float = 0.20
    requires_tau: bool = False

    #: Вес LDA-обмена сверх того, что уже входит в B88.
    lda_exchange_fraction: float = 0.08
    #: Вес обмена B88 (вместе с его LDA-частью).
    dft_exchange_fraction: float = 0.72
    #: Вес корреляции LYP; остаток до единицы — VWN(RPA).
    lyp_fraction: float = 0.81

    def __init__(self) -> None:
        """Собирает четыре полунелокальные части."""
        self._lda = LdaExchange()
        self._becke = BeckeExchange()
        self._lyp = LypCorrelation()
        self._vwn = VwnRpaCorrelation()

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Полунелокальная часть B3LYP; точный обмен подставляет решатель."""
        if spin_polarized:
            msg = "Спин-поляризованное вычисление идёт через evaluate_spin."
            raise ValueError(msg)
        lda = self._lda.evaluate(points, density)
        becke = self._becke.evaluate(points, density, density_gradient)
        lyp = self._lyp.evaluate(points, density, density_gradient)
        vwn = self._vwn.evaluate(points, density)

        w_lda, w_becke, w_lyp = (
            self.lda_exchange_fraction,
            self.dft_exchange_fraction,
            self.lyp_fraction,
        )
        w_vwn = 1.0 - w_lyp

        vsigma = None
        if becke.vsigma is not None:
            vsigma = w_becke * becke.vsigma + w_lyp * (
                lyp.vsigma if lyp.vsigma is not None else 0.0
            )
        return XcEvaluation(
            energy_density=(
                w_lda * lda.energy_density
                + w_becke * becke.energy_density
                + w_lyp * lyp.energy_density
                + w_vwn * vwn.energy_density
            ),
            vrho=(w_lda * lda.vrho + w_becke * becke.vrho + w_lyp * lyp.vrho + w_vwn * vwn.vrho),
            vsigma=vsigma,
        )

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Полунелокальная часть B3LYP для UKS; точный обмен подставляет решатель."""
        return _sum_spin_evaluations(
            points,
            density_spin,
            density_gradient_spin,
            (
                (self._lda, self.lda_exchange_fraction),
                (self._becke, self.dft_exchange_fraction),
                (self._lyp, self.lyp_fraction),
                (self._vwn, 1.0 - self.lyp_fraction),
            ),
        )


#: Функционалы, которые ядро действительно умеет считать. Реестр обращается к
#: этому словарю, поэтому «заявлено» и «реализовано» не могут разойтись.
FUNCTIONALS: dict[str, type[Svwn] | type[Pbe] | type[Pbe0] | type[Blyp] | type[B3lyp]] = {
    "svwn": Svwn,
    "lda": Svwn,
    "pbe": Pbe,
    "blyp": Blyp,
    "pbe0": Pbe0,
    "b3lyp": B3lyp,
}


def get_functional(name: str) -> Svwn | Pbe | Pbe0 | Blyp | B3lyp:
    """Возвращает реализованный функционал по имени.

    Бросает ``FunctionalNotFoundError`` — ту же ошибку, что и реестр
    возможностей. Два разных типа для одной и той же недоступности означали бы
    два разных сообщения пользователю в зависимости от пути вызова.
    """
    implementation = FUNCTIONALS.get(name)
    if implementation is None:
        raise FunctionalNotFoundError(name)
    return implementation()
