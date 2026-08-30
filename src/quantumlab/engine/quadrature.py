"""Квадратурная сетка для численного интегрирования обменного функционала.

DFT-обмен и корреляция — нелинейные функции плотности, поэтому интеграл
``E_xc = ∫ ρ ε_xc(ρ) dr`` берётся численно. Точность сетки напрямую влияет на
энергию, поэтому каждая часть проверяется независимо от остального кода:

* радиальная сетка — Гаусса–Чебышёва второго рода с отображением
  Мюррея–Хенда–Лэминга на ``[0, ∞)``;
* угловая сетка — произведение Гаусса–Лежандра по ``cos θ`` и равномерной сетки
  по ``φ``. Это менее эффективно, чем сетки Лебедева, но строится из первых
  принципов и интегрирует сферические гармоники точно до заданного порядка;
* разбиение Бекке (1988) — атомные веса, гладкие и дающие единицу в сумме.

Лебедевские сетки сознательно не используются: их узлы — табулированные корни,
которые пришлось бы либо зашить таблицей без вывода, либо взять из внешнего
пакета. Ни то, ни другое несовместимо с ADR-002 (внешние пакеты — только для
проверки, не источник истины).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import GridPreset
from quantumlab.engine.constants import angstrom_to_bohr

#: Число радиальных и угловых точек для каждого пресета сетки.
#: Подбирается так, чтобы ``coarse`` годилась для быстрых оценок, а
#: ``ultrafine`` — для значений, которые попадут в верификационный набор.
_GRID_POINTS: dict[GridPreset, tuple[int, int, int]] = {
    GridPreset.COARSE: (50, 12, 24),
    GridPreset.FINE: (75, 24, 48),
    GridPreset.ULTRAFINE: (99, 32, 64),
}


@dataclass(frozen=True, slots=True)
class QuadratureGrid:
    """Точки сетки и их веса.

    Attributes:
        points: координаты точек в борах, форма ``(n_points, 3)``.
        weights: полные веса (радиальный × угловой × вес Бекке), ``(n_points,)``.
        atom_index: индекс атома, к ячейке которого относится точка.
        preset: пресет плотности, которым сетка построена.
    """

    points: np.ndarray
    weights: np.ndarray
    atom_index: np.ndarray
    #: Пресет хранится вместе с сеткой: результат DFT без указания плотности
    #: сетки невоспроизводим, а спрашивать его у вызывающего кода — значит
    #: доверять, что тот не пересобрал сетку по-своему.
    preset: GridPreset = GridPreset.FINE

    @property
    def n_points(self) -> int:
        """Число точек сетки."""
        return int(self.weights.size)


def radial_grid(n_points: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Радиальная сетка Гаусса–Чебышёва второго рода на ``[0, ∞)``.

    Узлы квадратуры Гаусса–Чебышёва второго рода на ``[-1, 1]``:
    ``x_i = cos θ_i``, ``θ_i = iπ/(n+1)``, веса ``π sin²θ_i/(n+1)``. Отображение
    ``r = α(1+x)/(1-x)`` переводит отрезок в положительную полуось, а якобиан
    ``dr = 2α dx/(1-x)²`` вместе с весом Чебышёва даёт итоговый вес.

    Параметр ``α`` задаёт масштаб: его берут порядка размера атома, чтобы узлы
    гуще ложились там, где плотность велика.
    """
    index = np.arange(1, n_points + 1, dtype=float)
    theta = index * np.pi / (n_points + 1)
    x = np.cos(theta)
    radius = alpha * (1.0 + x) / (1.0 - x)
    weights = 2.0 * alpha * np.pi / (n_points + 1) * np.sin(theta) / (1.0 - x) ** 2
    # Отображение Мюррея–Хэнда–Лэминга даёт узлы по убыванию r; разворачиваем
    # в возрастающий порядок, чтобы сетку можно было читать как таблицу.
    # На сумму квадратуры порядок не влияет.
    return radius[::-1].copy(), weights[::-1].copy()


def angular_grid(n_theta: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Угловая сетка: Гаусс–Лежандр по ``cos θ`` и равномерная по ``φ``.

    Возвращает единичные векторы направлений и веса, сумма которых равна 4π.
    """
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0 * np.pi * np.arange(n_phi, dtype=float) / n_phi
    w_phi = np.full(n_phi, 2.0 * np.pi / n_phi)

    theta_grid, phi_grid = np.meshgrid(cos_theta, phi, indexing="ij")
    theta_w, phi_w = np.meshgrid(w_theta, w_phi, indexing="ij")
    weights = (theta_w * phi_w).ravel()

    sin_t = np.sqrt(np.clip(1.0 - theta_grid**2, 0.0, None)).ravel()
    directions = np.column_stack(
        (
            sin_t * np.cos(phi_grid.ravel()),
            sin_t * np.sin(phi_grid.ravel()),
            theta_grid.ravel(),
        )
    )
    return directions, weights


def becke_weights(
    points: np.ndarray, atom_centers: np.ndarray, atom_index: np.ndarray
) -> np.ndarray:
    """Веса Бекке (1988): гладкое разбиение единицы между атомами.

    Для каждой точки считается ``μ_ij = (r_i − r_j)/R_ij`` — смещение к атому
    ``i`` или от него, прогоняется через сглаживающий многочлен Бекке
    (трижды применённое ``p(x) = 1.5x − 0.5x³``), а затем нормируется так,
    чтобы веса всех атомов в точке давали в сумме единицу. Нормировка —
    не формальность: именно она делает разбиение точным.
    """
    n_atoms = atom_centers.shape[0]
    if n_atoms == 1:
        return np.ones(points.shape[0])

    # Расстояния от точек до каждого атома: (n_points, n_atoms).
    difference = points[:, None, :] - atom_centers[None, :, :]
    distances = np.linalg.norm(difference, axis=2)

    # Попарные расстояния между ядрами: (n_atoms, n_atoms).
    nucleus_difference = atom_centers[:, None, :] - atom_centers[None, :, :]
    nucleus_distances = np.linalg.norm(nucleus_difference, axis=2)
    np.fill_diagonal(nucleus_distances, 1.0)

    partition = np.ones((points.shape[0], n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            mu = (distances[:, i] - distances[:, j]) / nucleus_distances[i, j]
            partition[:, i] *= 0.5 * (1.0 - _becke_polynomial(mu))

    total = partition.sum(axis=1)
    return np.asarray(partition[np.arange(points.shape[0]), atom_index] / total)


def _becke_polynomial(mu: np.ndarray) -> np.ndarray:
    """Сглаживающий многочлен Бекке: ``p₃`` от ``1.5x − 0.5x³``.

    Тройное применение нужно, чтобы функция имела нулевые первую и вторую
    производные в точках ``±1`` — тогда веса стыкуются гладко и интеграл не
    зависит от того, где именно проходит граница между атомами.
    """
    value = 1.5 * mu - 0.5 * mu**3
    for _ in range(2):
        value = 1.5 * value - 0.5 * value**3
    return value


def build_grid(
    molecule: Molecule, preset: GridPreset = GridPreset.FINE, alpha: float = 0.8
) -> QuadratureGrid:
    """Собирает сетку для молекулы: радиальная × угловая × Бекке.

    Каждый атом получает свою радиальную сетку и полный набор углов; точки,
    оказавшиеся вдали от своего атома, всё равно учитываются — их вклад
    обнуляется весом Бекке, а не отбрасывается.
    """
    n_radial, n_theta, n_phi = _GRID_POINTS[preset]
    centers = np.array(
        [[angstrom_to_bohr(value) for value in atom.position] for atom in molecule.atoms]
    )

    radii, radial_weights = radial_grid(n_radial, alpha)
    directions, angular_weights = angular_grid(n_theta, n_phi)

    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    atom_index: list[np.ndarray] = []
    for index, center in enumerate(centers):
        # Точки сферического слоя: центр + r × направление.
        offset = radii[:, None, None] * directions[None, :, :]
        layer = center[None, None, :] + offset
        # Элемент объёма dV = r² dr dΩ: угловые веса уже содержат sin θ и дают
        # 4π, радиальные — меру dr, поэтому якобиан r² входит сюда явно. Без
        # него сетка интегрирует с весом, заниженным в r² раз, и плотность
        # оказывается «считанной» неверно.
        layer_weights = (radial_weights * radii**2)[:, None] * angular_weights[None, :]
        points.append(layer.reshape(-1, 3))
        weights.append(layer_weights.ravel())
        atom_index.append(np.full(layer_weights.size, index, dtype=int))

    all_points = np.vstack(points)
    all_weights = np.concatenate(weights)
    all_atoms = np.concatenate(atom_index)
    becke = becke_weights(all_points, centers, all_atoms)
    return QuadratureGrid(
        points=all_points,
        weights=all_weights * becke,
        atom_index=all_atoms,
        preset=preset,
    )
