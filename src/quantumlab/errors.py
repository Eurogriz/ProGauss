"""Человеко-понятная таксономия ошибок (§19 ТЗ).

Каждая ошибка — это не строка из логов, а **структурированный диагноз** из
трёх частей, которые GUI и CLI рендерят одинаково:

1. *Что произошло* — формулировка на человеческом языке, без кодов;
2. *Что мы попробовали* — список реально применённых алгоритмических стратегий;
3. *Что можно сделать* — набор исполняемых действий (кнопок), а не совет «попробуйте ещё раз».

Все тексты берутся из каталога переводов, поэтому ошибка автоматически
локализуется: ``scf.not_converged`` → ключи ``error.scf.not_converged.*``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from quantumlab.i18n import DEFAULT_LOCALE, has_message, t


class ErrorCode(StrEnum):
    """Стабильные коды ошибок.

    Код входит в контракт REST API (``ProblemDetails.code``) и в журнал
    аудита, поэтому значения нельзя переименовывать без повышения версии API.
    """

    METHOD_NOT_AVAILABLE = "engine.method_not_available"
    BASIS_NOT_FOUND = "registry.basis_not_found"
    FUNCTIONAL_NOT_FOUND = "registry.functional_not_found"
    MOLECULE_EMPTY = "molecule.empty"
    UNKNOWN_ELEMENT = "molecule.unknown_element"
    INVALID_MULTIPLICITY = "molecule.invalid_multiplicity"
    VALENCE_VIOLATION = "molecule.valence_violation"
    INVALID_JOB_TRANSITION = "job.invalid_transition"
    JOB_NOT_RESUMABLE = "job.not_resumable"
    SCF_NOT_CONVERGED = "scf.not_converged"
    OPTIMIZATION_NOT_CONVERGED = "optimization.not_converged"
    IMAGINARY_FREQUENCIES = "frequencies.imaginary_modes"
    ARTIFACT_MISSING = "storage.artifact_missing"
    OUT_OF_MEMORY = "runtime.out_of_memory"


class ActionKind(StrEnum):
    """Класс действия в блоке «Что можно сделать».

    * ``AUTOMATIC`` — система сама выполнит повторный расчёт с новыми настройками;
    * ``MANUAL``    — открывает форму настроек;
    * ``NAVIGATE``  — переводит пользователя в другой раздел (визуализация, лог).
    """

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    NAVIGATE = "navigate"


@dataclass(frozen=True, slots=True)
class DiagnosticAction:
    """Исполняемое действие, которое UI показывает кнопкой.

    Attributes:
        action_key: ключ локализации подписи (``action.*``).
        kind: класс действия.
        payload: машинные параметры для обработчика (например, профиль SCF).
    """

    action_key: str
    kind: ActionKind
    payload: Mapping[str, str] = field(default_factory=dict)


class QuantumLabError(Exception):
    """Базовое исключение с локализуемым диагнозом.

    Подкласс обязан задать :attr:`code`. Параметры подставляются в тексты
    каталога, поэтому один и тот же код ошибки даёт осмысленное сообщение
    на любом поддерживаемом языке.
    """

    code: ErrorCode

    def __init__(
        self,
        params: Mapping[str, object] | None = None,
        *,
        hint_params: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт ошибку с параметрами подстановки для текстов каталога."""
        self.params: dict[str, object] = dict(params or {})
        self.hint_params: dict[str, object] = dict(hint_params or {})
        super().__init__(self.explain())

    # -- ключи каталога ----------------------------------------------------- #
    @property
    def _prefix(self) -> str:
        return f"error.{self.code.value}"

    def title(self, locale: str = DEFAULT_LOCALE) -> str:
        """Короткий заголовок ошибки (первая строка в UI)."""
        return t(f"{self._prefix}.title", locale, **self.params)

    def what_happened(self, locale: str = DEFAULT_LOCALE) -> str:
        """Раздел «Что произошло»."""
        return t(f"{self._prefix}.what", locale, **self.params)

    def hint(self, locale: str = DEFAULT_LOCALE) -> str:
        """Пояснение/совет; пустая строка, если для кода не задана."""
        key = f"{self._prefix}.hint"
        if not has_message(key, locale):
            return ""
        return t(key, locale, **self.hint_params)

    def explain(self, locale: str = DEFAULT_LOCALE) -> str:
        """Полный текст диагноза для логов и CLI (без интерактивных кнопок).

        Первая строка — заголовок: так журнал расчёта остаётся читаемым, даже
        если UI показывает блоки отдельно.
        """
        parts = [self.title(locale), self.what_happened(locale)]
        suggestion = self.hint(locale)
        if suggestion:
            parts.append(suggestion)
        return "\n".join(parts)


class DiagnosisError(QuantumLabError):
    """Ошибка, у которой есть история предпринятых попыток и набор действий.

    Именно этот класс реализует сценарий из §19 ТЗ: SCF не сошёлся → показываем,
    какие стратегии уже были испробованы, и предлагаем конкретные следующие шаги.
    """

    def __init__(
        self,
        params: Mapping[str, object] | None = None,
        *,
        attempts: Iterable[str] = (),
        actions: Iterable[DiagnosticAction] = (),
        hint_params: Mapping[str, object] | None = None,
    ) -> None:
        """Создаёт диагноз.

        Args:
            params: параметры текстов «что произошло».
            attempts: ключи локализации применённых стратегий (``attempt.*``).
            actions: доступные пользователю действия.
            hint_params: параметры подсказки.
        """
        self.attempts: tuple[str, ...] = tuple(attempts)
        self.actions: tuple[DiagnosticAction, ...] = tuple(actions)
        super().__init__(params, hint_params=hint_params)

    def attempts_text(self, locale: str = DEFAULT_LOCALE) -> list[str]:
        """Локализованные названия испробованных стратегий."""
        return [t(key, locale) for key in self.attempts]

    def explain(self, locale: str = DEFAULT_LOCALE) -> str:
        """Текстовое представление всех трёх разделов (для CLI и логов)."""
        lines = [
            t("error.header", locale),
            self.title(locale),
            f"{t('error.section.what', locale)}: {self.what_happened(locale)}",
        ]
        hint = self.hint(locale)
        if hint:
            lines.append(hint)
        tried = self.attempts_text(locale)
        if tried:
            lines.append(f"{t('error.section.tried', locale)}: " + "; ".join(tried))
        if self.actions:
            labels = [t(action.action_key, locale) for action in self.actions]
            lines.append(f"{t('error.section.actions', locale)}: " + "; ".join(labels))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Конкретные ошибки. Каждая — отдельный тип, чтобы вызывающий код мог ловить
# именно то, что умеет обработать (например, GUI предлагает «найти ТС» только
# при ImaginaryFrequenciesError).
# --------------------------------------------------------------------------- #
class MethodNotAvailableError(QuantumLabError):
    """Метод объявлен в архитектуре, но не реализован (§54 ТЗ).

    Система обязана честно сообщать о нереализованном методе вместо того,
    чтобы выдавать приближение под видом точного результата.
    """

    code = ErrorCode.METHOD_NOT_AVAILABLE

    def __init__(self, method: str) -> None:
        """Создаёт ошибку для метода ``method``."""
        super().__init__({"method": method})
        self.method = method


class BasisNotFoundError(QuantumLabError):
    """Базисный набор отсутствует в реестре."""

    code = ErrorCode.BASIS_NOT_FOUND

    def __init__(self, basis: str) -> None:
        """Создаёт ошибку для базиса ``basis``."""
        super().__init__({"basis": basis})
        self.basis = basis


class FunctionalNotFoundError(QuantumLabError):
    """Функционал отсутствует в реестре."""

    code = ErrorCode.FUNCTIONAL_NOT_FOUND

    def __init__(self, functional: str) -> None:
        """Создаёт ошибку для функционала ``functional``."""
        super().__init__({"functional": functional})
        self.functional = functional


class EmptyMoleculeError(QuantumLabError):
    """Структура не содержит атомов."""

    code = ErrorCode.MOLECULE_EMPTY


class UnknownElementError(QuantumLabError):
    """Символ элемента не распознан."""

    code = ErrorCode.UNKNOWN_ELEMENT

    def __init__(self, symbol: str) -> None:
        """Создаёт ошибку для нераспознанного символа ``symbol``."""
        super().__init__({"symbol": symbol})
        self.symbol = symbol


class InvalidMultiplicityError(QuantumLabError):
    """Заряд и мультиплетность несовместимы по числу электронов."""

    code = ErrorCode.INVALID_MULTIPLICITY

    def __init__(
        self, *, charge: int, electrons: int, multiplicity: int, allowed: Iterable[int]
    ) -> None:
        """Создаёт ошибку и сразу считает список допустимых мультиплетностей."""
        allowed_tuple = tuple(allowed)
        super().__init__(
            {
                "charge": charge,
                "electrons": electrons,
                "multiplicity": multiplicity,
            },
            hint_params={"allowed": ", ".join(str(value) for value in allowed_tuple)},
        )
        self.charge = charge
        self.electrons = electrons
        self.multiplicity = multiplicity
        self.allowed: tuple[int, ...] = allowed_tuple


class ValenceViolationError(QuantumLabError):
    """Насчитанное число связей атома противоречит типичной валентности."""

    code = ErrorCode.VALENCE_VIOLATION

    def __init__(self, *, symbol: str, index: int, actual: int, expected: int) -> None:
        """Создаёт ошибку валентности для конкретного атома."""
        super().__init__({"symbol": symbol, "index": index, "actual": actual, "expected": expected})
        self.symbol = symbol
        self.index = index
        self.actual = actual
        self.expected = expected


class InvalidJobTransitionError(QuantumLabError):
    """Запрошен недопустимый переход состояния задания (§13 ТЗ)."""

    code = ErrorCode.INVALID_JOB_TRANSITION

    def __init__(self, current: str, target: str) -> None:
        """Создаёт ошибку перехода ``current`` → ``target``."""
        super().__init__({"current": current, "target": target})
        self.current = current
        self.target = target


class JobNotResumableError(QuantumLabError):
    """Для задания нет контрольной точки (§14 ТЗ)."""

    code = ErrorCode.JOB_NOT_RESUMABLE

    def __init__(self, job_id: str) -> None:
        """Создаёт ошибку для задания ``job_id``."""
        super().__init__({"job": job_id})
        self.job_id = job_id


class ScfNotConvergedError(DiagnosisError):
    """SCF не сошёлся; содержит историю стратегий и предлагаемые действия."""

    code = ErrorCode.SCF_NOT_CONVERGED

    def __init__(
        self,
        *,
        iterations: int,
        residual: float,
        attempts: Iterable[str] = (),
        actions: Iterable[DiagnosticAction] = (),
    ) -> None:
        """Создаёт диагноз несходимости SCF."""
        super().__init__(
            {"iterations": iterations, "residual": f"{residual:.3e}"},
            attempts=attempts,
            actions=actions,
        )
        self.iterations = iterations
        self.residual = residual


class OptimizationNotConvergedError(DiagnosisError):
    """Оптимизация геометрии не достигла порога по градиенту/смещению."""

    code = ErrorCode.OPTIMIZATION_NOT_CONVERGED

    def __init__(
        self,
        *,
        steps: int,
        max_gradient: float,
        threshold: float,
        attempts: Iterable[str] = (),
        actions: Iterable[DiagnosticAction] = (),
    ) -> None:
        """Создаёт диагноз несходимости оптимизации геометрии."""
        super().__init__(
            {
                "steps": steps,
                "max_gradient": f"{max_gradient:.3e}",
                "threshold": f"{threshold:.1e}",
            },
            attempts=attempts,
            actions=actions,
        )
        self.steps = steps
        self.max_gradient = max_gradient
        self.threshold = threshold


class ImaginaryFrequenciesError(DiagnosisError):
    """Найдены мнимые частоты — структура не является минимумом."""

    code = ErrorCode.IMAGINARY_FREQUENCIES

    def __init__(
        self,
        *,
        count: int,
        lowest: float,
        actions: Iterable[DiagnosticAction] = (),
    ) -> None:
        """Создаёт диагноз по найденным мнимым частотам."""
        super().__init__(
            {"count": count, "lowest": f"{lowest:.1f}"},
            actions=actions,
        )
        self.count = count
        self.lowest = lowest


class ArtifactMissingError(QuantumLabError):
    """Артефакт научных данных не найден в хранилище (§24 ТЗ)."""

    code = ErrorCode.ARTIFACT_MISSING

    def __init__(self, artifact: str) -> None:
        """Создаёт ошибку для отсутствующего артефакта ``artifact``."""
        super().__init__({"artifact": artifact})
        self.artifact = artifact


class OutOfMemoryError(DiagnosisError):
    """Процесс убит по памяти; предлагает конкретные способы снизить потребление."""

    code = ErrorCode.OUT_OF_MEMORY

    def __init__(
        self,
        *,
        requested: str,
        available: str,
        actions: Iterable[DiagnosticAction] = (),
    ) -> None:
        """Создаёт диагноз нехватки памяти с предложениями по её снижению."""
        super().__init__(
            {"requested": requested, "available": available},
            actions=actions,
        )
        self.requested = requested
        self.available = available
