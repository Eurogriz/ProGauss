"""Интерфейсы расчётного ядра (§23, §42, §43 ТЗ).

Здесь определены границы модулей из архитектурной схемы::

    integral-engine ─┐
    scf-engine ──────┼──► QuantumEngine (фасад) ──► CalculationResult
    dft-engine ──────┤
    correlation ─────┤
    optimization ────┤
    frequency ───────┘

Правила проектирования контрактов:

1. **Никаких зависимостей между модулями** — только общие типы данных и эти
   протоколы. ``scf-engine`` не знает, кто считает интегралы: NumPy-референс,
   C++/OpenMP или CUDA.
2. **Массивы — через явный алиас** :data:`Array` (``float64``). Это единственное
   место, где фиксируется точность по умолчанию; mixed precision (§7 ТЗ)
   вводится на уровне backend'а, а не контракта.
3. **Честность**: движок обязан вернуть ``ScfResult.converged = False`` вместо
   того, чтобы выдать несходящийся результат за сошедшийся.
4. **Прогресс и прерывание**: длительные операции принимают
   :class:`ProgressReporter`, который также служит точкой кооперативной отмены
   и записи чекпоинта.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.result import CalculationResult
from quantumlab.domain.spec import CalculationSpec

#: Матричный тип по умолчанию: dense ``float64`` в C-порядке.
Array = npt.NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Данные, передаваемые через границы модулей
# --------------------------------------------------------------------------- #
class ScfResult(BaseModel):
    """Результат SCF-процедуры.

    ``strategies_used`` — фактически применённые стратегии (DIIS, damping, …).
    Они попадают в журнал и в диагноз :class:`ScfNotConvergedError`, поэтому
    сообщение «Что мы попробовали» всегда описывает реальные действия движка,
    а не заранее написанный текст.
    """

    model_config = ConfigDict(extra="forbid")

    energy_hartree: float
    converged: bool
    iterations: int = Field(ge=0)
    energy_residual: float = Field(ge=0.0)
    density_residual: float = Field(ge=0.0)
    strategies_used: tuple[str, ...] = ()
    homo_energy_hartree: float | None = None
    lumo_energy_hartree: float | None = None
    dipole_debye: float | None = None


class EngineRequest(BaseModel):
    """Запрос к ядру: структура + спецификация + ресурсы.

    Специально не содержит ``Job``: ядро не должно знать про очередь,
    пользователей и хранилище — это задача Job Manager'а.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(
        description=(
            "Непрозрачный идентификатор задания. Ядро не знает про очередь, "
            "пользователей и хранилище — идентификатор нужен только потому, что "
            "CalculationResult обязан ссылаться на задание, породившее его."
        )
    )
    molecule: Molecule
    spec: CalculationSpec
    checkpoint: str | None = Field(
        default=None,
        description=(
            "Содержимое контрольной точки для рестарта SCF. Ядро не знает, "
            "откуда она взялась: хранилище — забота Job Manager'а."
        ),
    )
    threads: int = Field(default=1, ge=1)
    memory_mb: int = Field(default=2048, ge=1)
    resume_from: str | None = Field(default=None, description="URI чекпоинта для рестарта")


class CheckpointHandle(BaseModel):
    """Ссылка на сохранённое состояние расчёта (§14 ТЗ)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    iteration: int = Field(ge=0)
    sha256: str


class BackendCapabilities(BaseModel):
    """Что умеет конкретный вычислительный backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = Field(description="cpu | cuda | rocm")
    supports_gpu: bool = False
    supports_mixed_precision: bool = False
    supports_distributed: bool = False
    max_threads: int = Field(default=1, ge=1)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Протоколы (structural typing: реализация не обязана наследоваться)
# --------------------------------------------------------------------------- #
@runtime_checkable
class ProgressReporter(Protocol):
    """Приёмник прогресса; возвращает ``False``, если запрошена отмена."""

    def report(self, percent: float, stage_key: str, **extra: object) -> bool:
        """Сообщает о прогрессе и возвращает признак продолжения работы."""
        ...


@runtime_checkable
class ComputeBackend(Protocol):
    """Абстракция вычислительного устройства (CPU/CUDA/ROCm, §7 ТЗ).

    Backend отвечает за плотную линейную алгебру: диагонализацию, умножения,
    обратные матрицы. Физика (интегралы, функционалы) живёт выше и от backend'а
    не зависит, поэтому перенос на GPU не переписывает научный код.
    """

    @property
    def name(self) -> str:
        """Имя backend'а (попадает в журнал и отчёт)."""
        ...

    @property
    def capabilities(self) -> BackendCapabilities:
        """Возможности устройства."""
        ...

    def eigensolve_symmetric(
        self, matrix: Array, n_eigen: int | None = None
    ) -> tuple[Array, Array]:
        """Собственные значения/векторы симметричной матрицы (по возрастанию)."""
        ...

    def gemm(
        self, a: Array, b: Array, *, transpose_a: bool = False, transpose_b: bool = False
    ) -> Array:
        """Умножение матриц (BLAS level 3)."""
        ...


@runtime_checkable
class IntegralEngine(Protocol):
    """Одно- и двухэлектронные интегралы в базисе гауссиан.

    Сложность: однoэлектронные интегралы — O(N²), двухэлектронные — O(N⁴)
    без скрининга. Именно поэтому контракт сразу предусматривает
    ``screening_threshold`` и density-fitting путь (``fit_metric``), а не
    добавляет их позже (§7 ТЗ).
    """

    @property
    def name(self) -> str:
        """Имя реализации."""
        ...

    def overlap(self, molecule: Molecule, basis: str) -> Array:
        """Матрица перекрывания S."""
        ...

    def kinetic(self, molecule: Molecule, basis: str) -> Array:
        """Матрица кинетической энергии T."""
        ...

    def nuclear_attraction(self, molecule: Molecule, basis: str) -> Array:
        """Матрица притяжения электронов к ядрам V."""
        ...

    def electron_repulsion(
        self, molecule: Molecule, basis: str, *, screening_threshold: float = 1e-12
    ) -> Array:
        """Тензор (μν|λσ) либо сжатое представление — решает реализация."""
        ...

    def fit_metric(self, molecule: Molecule, aux_basis: str) -> Array:
        """Метрика вспомогательного базиса для RI/density fitting."""
        ...


@dataclass(frozen=True, slots=True)
class XcEvaluation:
    """Обменно-корреляционная энергия и потенциалы в точках квадратурной сетки.

    Возвращаем структуру, а не пару массивов: у LDA потенциал один, у GGA к
    нему добавляется производная по ``σ = |∇ρ|²``, у meta-GGA — по кинетической
    плотности ``τ``. Кортеж переменной длины заставил бы каждый решатель
    угадывать, что именно вернул конкретный функционал, и молча ломаться на
    следующем классе.

    Attributes:
        energy_density: ``ε_xc(r)`` — энергия на один электрон в точке.
        vrho: ``∂(ρ ε_xc)/∂ρ``. Именно производная от ``ρ ε_xc``, а не от
            ``ε_xc``: в выражение для потенциала входит первая из них.
        vsigma: ``∂(ρ ε_xc)/∂σ``, ``None`` для LDA.
        vtau: ``∂(ρ ε_xc)/∂τ``, ``None`` для LDA и GGA.
    """

    energy_density: Array
    vrho: Array
    vsigma: Array | None = None
    vtau: Array | None = None


@dataclass(frozen=True, slots=True)
class XcEvaluationSpin:
    """Спиново-разделённая XC-энергия и потенциалы в точках сетки (UKS).

    Каналы — ось 0: ``[0]`` = α, ``[1]`` = β. Обозначения — те же, что у
    :class:`XcEvaluation`, но производные берутся от энергии **на единицу
    объёма** ``E_V(r) = (ρ^α + ρ^β)·ε_xc`` по соответствующей переменной:

    Attributes:
        energy_density: ``ε_xc(r)`` — энергия на один электрон (по полной
            плотности). Энергия считается как ``Σ_g w_g (ρ^α + ρ^β) ε_xc``;
            величина одна на оба канала, потому что ``ε_xc`` — функция полной
            системы, а не сумма двух независимых функционалов.
        vrho: ``(2, n_points)`` — ``∂E_V/∂ρ^σ``. В фокиан канала α входит
            строка ``[0]``, в β — строка ``[1]``.
        vsigma: ``(2, 2, n_points)`` — ``∂E_V/∂s_στ``, где
            ``s_στ = ∇ρ^σ·∇ρ^τ``; ``None`` для LDA. Диагональ — производные по
            собственным градиентам каналов, внедиагональ — по смешанным.
        vtau: ``∂E_V/∂τ`` — не используется пока (meta-GGA вне текущего среза).
    """

    energy_density: Array
    vrho: Array
    vsigma: Array | None = None
    vtau: Array | None = None


@runtime_checkable
class ExchangeCorrelationFunctional(Protocol):
    """Обменно-корреляционный функционал (§5 ТЗ).

    Расширяемость достигается тем, что функционал — это объект с декларативным
    описанием (класс, параметры, источник), а не ветка ``if`` внутри SCF.
    """

    @property
    def name(self) -> str:
        """Имя функционала, например ``pbe0``."""
        ...

    @property
    def functional_class(self) -> str:
        """Класс: lda / gga / mgga / hybrid / range_separated_hybrid / double_hybrid."""
        ...

    @property
    def is_hybrid(self) -> bool:
        """Содержит ли функционал долю точного обмена."""
        ...

    @property
    def exact_exchange_fraction(self) -> float:
        """Доля точного обмена (0 для LDA/GGA, 0.25 для PBE0, …)."""
        ...

    def evaluate(
        self,
        points: Array,
        density: Array,
        density_gradient: Array | None = None,
        *,
        spin_polarized: bool = False,
    ) -> XcEvaluation:
        """Вычисляет ``ε_xc`` и потенциалы в точках сетки.

        Args:
            points: координаты точек, ``(n_points, 3)``.
            density: плотность ``ρ`` в точках, ``(n_points,)``.
            density_gradient: ``∇ρ`` в точках, ``(n_points, 3)``. Обязателен для
                GGA и выше; LDA-функционал его игнорирует.
            spin_polarized: флаг сохранён для совместимости; спин-поляризованное
                вычисление идёт через :meth:`evaluate_spin`, и передача
                ``True`` отклоняется — иначе «раздельные каналы» молча
                перестанут быть различимы от полной плотности.
        """
        ...

    def evaluate_spin(
        self,
        points: Array,
        density_spin: Array,
        density_gradient_spin: Array | None = None,
    ) -> XcEvaluationSpin:
        """Спиново-разделённая версия :meth:`evaluate` для UKS.

        Args:
            points: координаты точек, ``(n_points, 3)``.
            density_spin: ``(2, n_points)`` — плотности каналов α и β.
            density_gradient_spin: ``(2, n_points, 3)`` — градиенты каналов.
                Обязателен для GGA; LDA-функционал игнорирует.
        """
        ...


@runtime_checkable
class ScfSolver(Protocol):
    """Итерационное решение уравнений Хартри–Фока / Кона–Шэма (§10 ТЗ)."""

    @property
    def name(self) -> str:
        """Имя решателя."""
        ...

    def solve(
        self,
        request: EngineRequest,
        *,
        integrals: IntegralEngine,
        functional: ExchangeCorrelationFunctional | None,
        backend: ComputeBackend,
        progress: ProgressReporter | None = None,
    ) -> ScfResult:
        """Выполняет SCF-цикл и возвращает результат с признаком сходимости."""
        ...


@runtime_checkable
class GradientEngine(Protocol):
    """Аналитические градиенты энергии по координатам ядер."""

    @property
    def name(self) -> str:
        """Имя реализации."""
        ...

    def gradient(self, request: EngineRequest, *, scf: ScfResult) -> Array:
        """Градиент энергии: массив формы ``(n_atoms, 3)``, Eh/Bohr."""
        ...


@runtime_checkable
class CorrelationEngine(Protocol):
    """Пост-Хартри–Фок корреляция: MP2, SCS-MP2, CCSD, CCSD(T) (§5 ТЗ).

    Интерфейс намеренно узкий: метод получает сошедшуюся SCF-точку и возвращает
    поправку к энергии, поэтому его можно добавить, не переписывая систему.
    """

    @property
    def name(self) -> str:
        """Имя метода, например ``mp2``."""
        ...

    def correlation_energy(self, request: EngineRequest, *, scf: ScfResult) -> float:
        """Корреляционная поправка к энергии, Eh."""
        ...


@runtime_checkable
class OptimizerEngine(Protocol):
    """Оптимизация геометрии (§11 ТЗ)."""

    @property
    def name(self) -> str:
        """Имя оптимизатора."""
        ...

    def optimize(
        self,
        request: EngineRequest,
        *,
        energy_and_gradient: Callable[[Molecule], tuple[float, Array]],
        progress: ProgressReporter | None = None,
    ) -> Molecule:
        """Оптимизирует геометрию, возвращая новую структуру."""
        ...


@runtime_checkable
class FrequencyEngine(Protocol):
    """Колебательные частоты и термодинамика (§9 ТЗ)."""

    @property
    def name(self) -> str:
        """Имя реализации."""
        ...

    def frequencies(
        self, request: EngineRequest, *, hessian: Array
    ) -> tuple[tuple[float, ...], Array]:
        """Частоты в см⁻¹ и матрица нормальных мод."""
        ...


@runtime_checkable
class PropertyEngine(Protocol):
    """Свойства: мультиполи, заряды, плотности, ESP (§9 ТЗ)."""

    @property
    def name(self) -> str:
        """Имя реализации."""
        ...

    def properties(self, request: EngineRequest, *, scf: ScfResult) -> Mapping[str, Any]:
        """Набор вычисленных свойств, ключи стабильны в пределах версии API."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """Хранилище контрольных точек (§14 ТЗ).

    Ключ — ``(job_id, attempt)``: перезапуск не затирает чекпоинт предыдущей
    попытки, что делает повторы идемпотентными.
    """

    def save(self, job_id: str, attempt: int, state: Mapping[str, Any]) -> CheckpointHandle:
        """Сохраняет состояние и возвращает ссылку на него."""
        ...

    def latest(self, job_id: str) -> CheckpointHandle | None:
        """Последний чекпоинт задания или ``None``."""
        ...

    def load(self, handle: CheckpointHandle) -> dict[str, Any]:
        """Восстанавливает состояние по ссылке."""
        ...


@runtime_checkable
class QuantumEngine(Protocol):
    """Фасад расчётного ядра — единственный вход для Job Manager'а.

    Все интерфейсы пользователя (GUI, CLI, REST, SDK) вызывают именно его,
    поэтому подключение внешнего пакета (§42 ТЗ) сводится к реализации этого
    протокола адаптером.
    """

    @property
    def name(self) -> str:
        """Имя ядра, например ``quantumlab-reference``."""
        ...

    @property
    def version(self) -> str:
        """Версия ядра — входит в отпечаток расчёта (§40 ТЗ)."""
        ...

    def supported_tasks(self) -> Sequence[str]:
        """Задачи, которые ядро действительно умеет выполнять."""
        ...

    def run(
        self, request: EngineRequest, *, progress: ProgressReporter | None = None
    ) -> CalculationResult:
        """Выполняет расчёт и возвращает результат с проверками качества."""
        ...
