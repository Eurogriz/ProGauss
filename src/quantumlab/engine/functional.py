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

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import BasisSet, cartesian_powers
from quantumlab.engine.contracts import Array, XcEvaluation
from quantumlab.errors import FunctionalNotFoundError

#: (3/π)^(1/3) — входит в обмен Слэтера.
_SLATER_FACTOR: float = (3.0 / np.pi) ** (1.0 / 3.0)

#: Порог плотности, ниже которого вклад считается нулевым. Без него деление на
#: ρ^(1/3) и логарифм r_s дают ``nan`` в хвостах плотности, и одна такая точка
#: отравляет всю матрицу Фока.
_DENSITY_FLOOR: float = 1e-14


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
        del points, density_gradient, spin_polarized  # LDA зависит только от ρ
        rho = np.asarray(density, dtype=float)
        safe = np.where(rho > _DENSITY_FLOOR, rho, 0.0)
        cube_root = np.cbrt(safe)
        exc = np.where(safe > 0.0, -0.75 * _SLATER_FACTOR * cube_root, 0.0)
        vrho = np.where(safe > 0.0, -_SLATER_FACTOR * cube_root, 0.0)
        return XcEvaluation(energy_density=np.asarray(exc), vrho=np.asarray(vrho))


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
    ) -> XcEvaluation:
        """Энергия и потенциал корреляционной части.

        Потенциал считается как ``v_c = ε_c − (r_s/3) dε_c/dr_s``, что
        эквивалентно ``d(ρ ε_c)/dρ`` через ``ρ ∝ r_s^{-3}``. Производная
        берётся аналитически, а не численно: конечная разность по ``r_s``
        в области малых плотностей дала бы шум, сравнимый с самой поправкой.
        """
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

    Это LDA-функционал: локальная плотность плотности, без градиентов. Для
    спиновой поляризации нужен UKS, которого пока нет, поэтому параметр
    ``spin_polarized`` принят только для соответствия протоколу и отклоняется.
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
            msg = (
                "Спиново-поляризованный XC требует UKS (отдельные плотности α и β), "
                "который не реализован."
            )
            raise NotImplementedError(msg)
        exchange = self._exchange.evaluate(points, density, density_gradient)
        correlation = self._correlation.evaluate(points, density, density_gradient)
        return XcEvaluation(
            energy_density=exchange.energy_density + correlation.energy_density,
            vrho=exchange.vrho + correlation.vrho,
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
    ) -> XcEvaluation:
        """Энергия и потенциал корреляции PW92."""
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
    ) -> XcEvaluation:
        """Энергия и потенциалы обмена PBE."""
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


class PbeCorrelation:
    """Корреляция PBE (GGA_C_PBE), неполяризованный случай.

    ``ε_c = ε_c^PW92 + H``, ``H = γ ln[1 + (β/γ)·(t² + A t⁴)/(1 + A t² + A² t⁴)]``

    ``A = (β/γ)/[exp(−ε_c^PW92/γ) − 1]``, ``t = |∇ρ|/(2 k_s ρ)``, ``k_s = √(4k_F/π)``.

    Для неполяризованного газа спиновый фактор ``φ = 1``. Производные по ``ρ``
    берутся с учётом того, что ``A`` тоже зависит от ``ρ`` — иначе потенциал
    расходится с энергией, и расхождение не видно ни по сходимости SCF, ни по
    коммутатору, только по энергии.
    """

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
    ) -> XcEvaluation:
        """Энергия и потенциалы корреляции PBE."""
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
            msg = "Спиново-поляризованный XC требует UKS (отдельные плотности α и β)."
            raise NotImplementedError(msg)
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
            msg = "Спиново-поляризованный XC требует UKS (отдельные плотности α и β)."
            raise NotImplementedError(msg)
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


#: Функционалы, которые ядро действительно умеет считать. Реестр обращается к
#: этому словарю, поэтому «заявлено» и «реализовано» не могут разойтись.
FUNCTIONALS: dict[str, type[Svwn] | type[Pbe] | type[Pbe0]] = {
    "svwn": Svwn,
    "lda": Svwn,
    "pbe": Pbe,
    "pbe0": Pbe0,
}


def get_functional(name: str) -> Svwn | Pbe | Pbe0:
    """Возвращает реализованный функционал по имени.

    Бросает ``FunctionalNotFoundError`` — ту же ошибку, что и реестр
    возможностей. Два разных типа для одной и той же недоступности означали бы
    два разных сообщения пользователю в зависимости от пути вызова.
    """
    implementation = FUNCTIONALS.get(name)
    if implementation is None:
        raise FunctionalNotFoundError(name)
    return implementation()
