"""Каталог переводов и механизм разрешения ключей.

Каталог — это плоский словарь ``ключ -> строка`` с именованными параметрами
(``{molecule}``). Параметризация обязательна: порядок слов в русском и
английском разный, поэтому конкатенация строк запрещена.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Any, cast

#: Поддерживаемые языки. Русский — первый и потому язык по умолчанию.
SUPPORTED_LOCALES: tuple[str, ...] = ("ru", "en")

#: Язык по умолчанию (§3 ТЗ).
DEFAULT_LOCALE: str = "ru"

#: Тип языка интерфейса.
Locale = str


class MissingTranslationKeyError(KeyError):
    """Ключ перевода отсутствует в каталоге.

    Это ошибка разработчика, а не пользователя: UI обязан показывать осмысленный
    текст. В production вместо падения можно включить fallback (см. ``strict``),
    но тесты паритета локализаций (``tests/test_i18n_parity.py``) не дают такому
    состоянию дожить до релиза.
    """

    def __init__(self, key: str, locale: str) -> None:
        """Запоминает ключ и локаль, в которой перевода не нашлось."""
        super().__init__(key)
        self.key = key
        self.locale = locale

    def __str__(self) -> str:
        """Человекочитаемое сообщение для разработчика."""
        return f"Нет перевода для ключа {self.key!r} в локали {self.locale!r}"


class Catalog:
    """Неизменяемый набор переводов для одной локали."""

    __slots__ = ("_locale", "_messages")

    def __init__(self, locale: str, messages: dict[str, str]) -> None:
        """Создаёт каталог из уже загруженного словаря строк."""
        self._locale = locale
        self._messages = dict(messages)

    @property
    def locale(self) -> str:
        """Код локали (``ru``, ``en``)."""
        return self._locale

    def keys(self) -> frozenset[str]:
        """Все ключи каталога — используется тестом паритета локализаций."""
        return frozenset(self._messages)

    def __contains__(self, key: object) -> bool:
        """Есть ли ключ в каталоге."""
        return key in self._messages

    def __len__(self) -> int:
        """Число ключей в каталоге."""
        return len(self._messages)

    def has(self, key: str) -> bool:
        """Есть ли перевод для ключа.

        Нужна для опциональных блоков (подсказки у ошибок определены не для
        всех кодов). Опираться на ``try_t`` здесь нельзя: он по контракту
        возвращает сам ключ, и вызывающий код не отличит «перевода нет» от
        «перевод совпадает с ключом».
        """
        return key in self._messages

    def get(
        self, key: str, params: Mapping[str, object] | None = None, *, strict: bool = True
    ) -> str:
        """Возвращает локализованную строку, подставляя именованные параметры.

        Параметры передаются словарём, а не именованными аргументами: имена
        плейсхолдеров задаются переводчиками и не должны конфликтовать со
        служебными аргументами функции.

        Args:
            key: dotted-ключ перевода.
            params: словарь подстановки для плейсхолдеров ``{name}``.
            strict: при ``True`` отсутствие ключа — ошибка. При ``False``
                возвращается сам ключ (диагностический режим).
        """
        template = self._messages.get(key)
        if template is None:
            if strict:
                raise MissingTranslationKeyError(key, self._locale)
            return key
        if not params:
            return template
        return template.format(**params)


@lru_cache(maxsize=len(SUPPORTED_LOCALES))
def get_catalog(locale: str = DEFAULT_LOCALE) -> Catalog:
    """Загружает и кэширует каталог для локали.

    Неизвестная локаль приводит к ``ValueError``: тихий откат на английский
    нарушил бы требование «русский — язык по умолчанию».
    """
    if locale not in SUPPORTED_LOCALES:
        msg = f"Локаль {locale!r} не поддерживается. Доступны: {', '.join(SUPPORTED_LOCALES)}"
        raise ValueError(msg)
    resource = resources.files("quantumlab.i18n.locales").joinpath(f"{locale}.json")
    raw: Any = json.loads(resource.read_text(encoding="utf-8"))
    messages = cast("dict[str, str]", raw)
    return Catalog(locale, messages)


def t(key: str, locale: str = DEFAULT_LOCALE, **params: object) -> str:
    """Короткая форма :func:`Catalog.get` — основная точка доступа из UI/CLI.

    Отсутствие ключа трактуется как ошибка разработчика: интерфейс обязан
    показывать осмысленный текст, а не ``error.scf.title``.
    """
    return get_catalog(locale).get(key, params)


def try_t(key: str, locale: str = DEFAULT_LOCALE, **params: object) -> str:
    """Как :func:`t`, но возвращает сам ключ, если перевода нет.

    Применяется только для опциональных блоков (например, подсказки у ошибок,
    которые определены не для всех кодов).
    """
    return get_catalog(locale).get(key, params, strict=False)


def has_message(key: str, locale: str = DEFAULT_LOCALE) -> bool:
    """Есть ли перевод для ключа в каталоге локали."""
    return get_catalog(locale).has(key)


def localize(
    key: str,
    locale: str,
    fallback: str = DEFAULT_LOCALE,
    params: Mapping[str, object] | None = None,
) -> str:
    """Перевод с откатом на ``fallback``.

    Используется в production-режиме: лучше показать русский текст, чем
    необработанный ключ. Тест паритета локализаций гарантирует, что откат
    не маскирует реальные пропуски.
    """
    catalog = get_catalog(locale)
    if key in catalog:
        return catalog.get(key, params)
    return get_catalog(fallback).get(key, params)
