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
from quantumlab.engine.contracts import Array
from quantumlab.errors import FunctionalNotFoundError

#: (3/π)^(1/3) — входит в обмен Слэтера.
_SLATER_FACTOR: float = (3.0 / np.pi) ** (1.0 / 3.0)

#: Порог плотности, ниже которого вклад считается нулевым. Без него деление на
#: ρ^(1/3) и логарифм r_s дают ``nan`` в хвостах плотности, и одна такая точка
#: отравляет всю матрицу Фока.
_DENSITY_FLOOR: float = 1e-14


def evaluate_basis(basis: BasisSet, molecule: Molecule, points: np.ndarray) -> np.ndarray:
    """Значения базисных функций в точках сетки.

    Возвращает массив ``(n_points, n_functions)``. Функция считается как сумма
    примитивов с учётом коэффициентов сжатия, норм примитивов и поправок на
    норму декартовой компоненты — ровно те же множители, что используются при
    сборке интегралов, поэтому плотность из сетки согласована с плотностью из
    матрицы.

    Центры берутся из молекулы: ``BasisSet`` хранит только индекс атома в
    оболочке, сами координаты живут в структуре.
    """
    from quantumlab.engine.constants import angstrom_to_bohr

    centers = np.array(
        [[angstrom_to_bohr(value) for value in atom.position] for atom in molecule.atoms]
    )
    n_points = points.shape[0]
    columns: list[np.ndarray] = []
    for shell in basis.shells:
        center = centers[shell.center]
        delta = points - center[None, :]
        distance_squared = np.sum(delta * delta, axis=1)
        radial = np.zeros((shell.n_primitives, n_points))
        for index, exponent in enumerate(shell.exponents):
            radial[index] = np.exp(-exponent * distance_squared)
        scales = shell.component_scales
        for component, powers in enumerate(cartesian_powers(shell.angular_momentum)):
            angular = np.ones(n_points)
            for axis, power in enumerate(powers):
                if power:
                    angular *= delta[:, axis] ** power
            contracted = shell.coefficients @ radial
            columns.append(angular * scales[component] * contracted)
    return np.column_stack(columns) if columns else np.zeros((n_points, 0))


def density_at_points(values: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Плотность ``ρ(r) = Σ_μν D_μν φ_μ φ_ν`` в точках сетки.

    Симметрия матрицы плотности используется явно: диагональ входит один раз,
    недиагональные элементы — вдвое.
    """
    diagonal = np.einsum("pg,g,pg->p", values, np.diag(density), values)
    off_diagonal = np.einsum("pg,gh,ph->p", values, density - np.diag(np.diag(density)), values)
    return np.asarray(diagonal + off_diagonal)


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
        self, points: Array, density: Array, *, spin_polarized: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает ``(exc, vxc)`` в точках сетки."""
        del points, spin_polarized  # LDA зависит только от величины плотности
        rho = np.asarray(density, dtype=float)
        safe = np.where(rho > _DENSITY_FLOOR, rho, 0.0)
        cube_root = np.cbrt(safe)
        exc = np.where(safe > 0.0, -0.75 * _SLATER_FACTOR * cube_root, 0.0)
        vxc = np.where(safe > 0.0, -_SLATER_FACTOR * cube_root, 0.0)
        return np.asarray(exc), np.asarray(vxc)


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
        self, points: Array, density: Array, *, spin_polarized: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает ``(exc, vxc)`` корреляционной части.

        Потенциал считается как ``v_c = ε_c − (r_s/3) dε_c/dr_s``, что
        эквивалентно ``d(ρ ε_c)/dρ`` через ``ρ ∝ r_s^{-3}``. Производная
        берётся аналитически, а не численно: конечная разность по ``r_s``
        в области малых плотностей дала бы шум, сравнимый с самой поправкой.
        """
        del points, spin_polarized
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
        vxc = np.zeros_like(rho)
        vxc[valid] = exc[valid] - x[valid] / 6.0 * derivative[valid]
        return exc, vxc

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
        self, points: np.ndarray, density: np.ndarray, *, spin_polarized: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Энергия и потенциал обмена+корреляции; см. протокол."""
        if spin_polarized:
            msg = (
                "Спиново-поляризованный XC требует UKS (отдельные плотности α и β), "
                "который не реализован."
            )
            raise NotImplementedError(msg)
        exchange_energy, exchange_potential = self._exchange.evaluate(points, density)
        correlation_energy, correlation_potential = self._correlation.evaluate(points, density)
        return exchange_energy + correlation_energy, exchange_potential + correlation_potential


#: Функционалы, которые ядро действительно умеет считать. Реестр обращается к
#: этому словарю, поэтому «заявлено» и «реализовано» не могут разойтись.
FUNCTIONALS: dict[str, type[Svwn]] = {"svwn": Svwn, "lda": Svwn}


def get_functional(name: str) -> Svwn:
    """Возвращает реализованный функционал по имени.

    Бросает ``FunctionalNotFoundError`` — ту же ошибку, что и реестр
    возможностей. Два разных типа для одной и той же недоступности означали бы
    два разных сообщения пользователю в зависимости от пути вызова.
    """
    implementation = FUNCTIONALS.get(name)
    if implementation is None:
        raise FunctionalNotFoundError(name)
    return implementation()
