"""Локализация QuantumLab.

Русский язык — язык интерфейса **по умолчанию** (§3 ТЗ), английский —
обязательный второй язык. Архитектура локализации заложена до реализации UI:
в коде нет ни одного жёстко зашитого UI-текста, есть только **ключи**,
которые разрешаются в строки каталога :class:`Catalog`.

Ключи имеют dotted-иерархию и сгруппированы по доменам::

    nav.projects                 элементы навигации
    wizard.step.molecule         шаги мастера расчёта
    status.queued                состояния задания
    error.<код>.title            человекочитаемые ошибки (§19 ТЗ)
    tooltip.<параметр>.what      объяснимость (§18 ТЗ)
    profile.<профиль>.name       профили точности (§8 ТЗ)
    capability.<метод>.notes     честное описание ограничений (§54 ТЗ)
"""

from quantumlab.i18n.catalog import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    Catalog,
    Locale,
    MissingTranslationKeyError,
    get_catalog,
    has_message,
    localize,
    t,
    try_t,
)

__all__ = [
    "DEFAULT_LOCALE",
    "SUPPORTED_LOCALES",
    "Catalog",
    "Locale",
    "MissingTranslationKeyError",
    "get_catalog",
    "has_message",
    "localize",
    "t",
    "try_t",
]
