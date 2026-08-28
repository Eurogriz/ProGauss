"""Описание возможностей системы (§41, §54 ТЗ).

Ключевое требование ТЗ: *система не имеет права выдавать нереализованный метод
за работающий*. Поэтому у каждой возможности есть явный статус
:class:`Availability`, и UI обязан показывать «Этот метод пока недоступен»
вместо того, чтобы молча подменять расчёт.

``Capability`` — также единица **плагинной системы**: внешний плагин
регистрирует такие же объекты, описывая, что он добавляет.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from quantumlab.i18n import DEFAULT_LOCALE, t


class CapabilityKind(StrEnum):
    """Категория возможности."""

    METHOD = "method"
    FUNCTIONAL = "functional"
    BASIS = "basis"
    TASK = "task"
    BACKEND = "backend"
    FORMAT = "format"
    SCHEDULER = "scheduler"
    PROPERTY = "property"
    COORDINATES = "coordinates"
    SPIN = "spin"
    DISPERSION = "dispersion"
    SCF = "scf"
    OPTIMIZER = "optimizer"


class Availability(StrEnum):
    """Статус реализации."""

    #: Реализовано и прошло верификационный набор.
    IMPLEMENTED = "implemented"
    #: Реализовано частично — ограничения описаны в ``limitations``.
    PARTIAL = "partial"
    #: Объявлено в архитектуре, но расчётный код отсутствует.
    NOT_IMPLEMENTED = "not_implemented"

    @property
    def is_usable(self) -> bool:
        """Можно ли выбирать эту возможность в расчёте."""
        return self is not Availability.NOT_IMPLEMENTED


@dataclass(frozen=True, slots=True)
class Capability:
    """Декларативное описание одной возможности системы.

    Attributes:
        id: стабильный идентификатор вида ``method:mp2``.
        kind: категория.
        name: машиночитаемое имя (то, что приходит из API/CLI).
        availability: статус реализации.
        since_version: версия, в которой появилось (для IMPLEMENTED/PARTIAL).
        limitations: ограничения для PARTIAL.
        metadata: дополнительные атрибуты (класс функционала, число функций и т. п.).
        aliases: альтернативные написания для пользовательского ввода.
        notes_key: ключ локализации пояснения.
    """

    id: str
    kind: CapabilityKind
    name: str
    availability: Availability
    since_version: str | None = None
    limitations: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    notes_key: str = "capability.note.not_implemented"

    @property
    def is_usable(self) -> bool:
        """Можно ли использовать в реальном расчёте."""
        return self.availability.is_usable

    def describe(self, locale: str = DEFAULT_LOCALE) -> str:
        """Локализованное пояснение статуса — для раздела «База методов»."""
        if self.availability is Availability.PARTIAL:
            return t(self.notes_key, locale, scope="; ".join(self.limitations))
        return t(self.notes_key, locale)
