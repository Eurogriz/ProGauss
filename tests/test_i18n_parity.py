"""Проверка локализации: паритет ru/en и полнота ключей.

Это не «тест ради теста»: именно он гарантирует выполнение требования §3 ТЗ
(русский интерфейс по умолчанию + обязательная английская локализация) и не
даёт добавиться в код UI-тексту, которого нет в каталоге.
"""

from __future__ import annotations

import re

import pytest

from quantumlab.domain.spec import PrecisionProfile, Task
from quantumlab.engine.reference import WARNING_KEYS
from quantumlab.errors import ErrorCode
from quantumlab.i18n import (
    DEFAULT_LOCALE,
    SUPPORTED_LOCALES,
    Catalog,
    MissingTranslationKeyError,
    get_catalog,
    localize,
    t,
    try_t,
)
from quantumlab.jobs.state_machine import JobStatus

PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def catalogs() -> dict[str, Catalog]:
    return {locale: get_catalog(locale) for locale in SUPPORTED_LOCALES}


def test_russian_is_default_locale() -> None:
    assert DEFAULT_LOCALE == "ru"
    assert SUPPORTED_LOCALES[0] == "ru"
    assert set(SUPPORTED_LOCALES) == {"ru", "en"}


def test_locales_have_identical_key_sets() -> None:
    loaded = catalogs()
    reference = loaded["ru"].keys()
    for locale, catalog in loaded.items():
        missing = reference - catalog.keys()
        extra = catalog.keys() - reference
        assert not missing, f"{locale}: не хватает ключей {sorted(missing)[:10]}"
        assert not extra, f"{locale}: лишние ключи {sorted(extra)[:10]}"


def test_no_empty_or_key_like_values() -> None:
    for locale, catalog in catalogs().items():
        for key in sorted(catalog.keys()):
            value = catalog.get(key)
            assert value.strip(), f"{locale}: пустое значение у {key}"
            assert value != key, f"{locale}: значение равно ключу {key}"


def test_placeholders_match_across_locales() -> None:
    loaded = catalogs()
    for key in sorted(loaded["ru"].keys()):
        expected = set(PLACEHOLDER.findall(loaded["ru"].get(key)))
        for locale in SUPPORTED_LOCALES:
            actual = set(PLACEHOLDER.findall(loaded[locale].get(key)))
            assert expected == actual, (
                f"{key}: параметры расходятся в {locale}: {expected} != {actual}"
            )


def test_unknown_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="не поддерживается"):
        get_catalog("de")


def test_missing_key_raises_in_strict_mode() -> None:
    with pytest.raises(MissingTranslationKeyError):
        t("no.such.key")
    assert try_t("no.such.key") == "no.such.key"


def test_localize_falls_back_to_russian() -> None:
    assert localize("app.name", "en") == "QuantumLab"
    assert localize("app.name", "ru") == "QuantumLab"


def test_every_error_code_is_fully_translated() -> None:
    loaded = catalogs()
    for code in ErrorCode:
        for suffix in ("title", "what"):
            key = f"error.{code.value}.{suffix}"
            for locale in SUPPORTED_LOCALES:
                assert key in loaded[locale], f"нет ключа {key} в {locale}"


def test_every_status_task_and_profile_is_translated() -> None:
    loaded = catalogs()
    required = [f"status.{status.value}" for status in JobStatus]
    required += [f"task.{task.value}.title" for task in Task]
    required += [f"task.{task.value}.description" for task in Task]
    required += [f"profile.{profile.value}.name" for profile in PrecisionProfile]
    required += [f"profile.{profile.value}.description" for profile in PrecisionProfile]
    for key in required:
        for locale in SUPPORTED_LOCALES:
            assert key in loaded[locale], f"нет ключа {key} в {locale}"


@pytest.mark.parametrize(
    "parameter",
    [
        "charge",
        "multiplicity",
        "functional",
        "basis",
        "dispersion",
        "grid",
        "scf_threshold",
        "integral_threshold",
        "threads",
        "memory",
        "device",
    ],
)
def test_explainability_tooltips_are_complete(parameter: str) -> None:
    """§18 ТЗ: у каждого сложного параметра — все четыре пояснения."""
    loaded = catalogs()
    for suffix in ("title", "what", "why", "if_changed", "relevant_for"):
        key = f"tooltip.{parameter}.{suffix}"
        for locale in SUPPORTED_LOCALES:
            assert key in loaded[locale], f"нет ключа {key} в {locale}"


def test_error_explanation_has_all_three_sections() -> None:
    """§19 ТЗ: что произошло / что попробовали / что можно сделать."""
    from quantumlab.errors import ActionKind, DiagnosticAction, ScfNotConvergedError

    error = ScfNotConvergedError(
        iterations=80,
        residual=3.1e-5,
        attempts=("attempt.diis", "attempt.damping", "attempt.level_shift"),
        actions=(
            DiagnosticAction("action.retry_automatic", ActionKind.AUTOMATIC),
            DiagnosticAction("action.use_robust_scf", ActionKind.AUTOMATIC),
            DiagnosticAction("action.edit_settings", ActionKind.MANUAL),
        ),
    )
    text = error.explain("ru")
    assert "Что произошло" in text
    assert "Что мы попробовали" in text
    assert "Что можно сделать" in text
    assert "DIIS-ускорение" in text
    assert "Повторить автоматически" in text
    assert "80" in text

    english = error.explain("en")
    assert "What happened" in english
    assert "Retry automatically" in english


def test_every_engine_warning_key_is_translated_in_both_locales() -> None:
    """Каждое предупреждение движка переводится на оба языка.

    Движок возвращает ключ, а не текст, поэтому пропущенный перевод вылез бы не
    в тесте, а у пользователя: ``t()`` бросает исключение, и интерфейс упал бы
    на предупреждении. Проверяем прямо: ключ обязан рендериться и подставлять
    параметры.
    """
    placeholders = {
        "warning.scf_not_converged": {"iterations": "12"},
        "warning.basis_spherical_scheme": {"basis": "cc-pvdz"},
        "warning.dipole_origin_charged": {"charge": "+1"},
        "warning.grid_prune_unimplemented": {},
        "warning.grid_xc_integration": {"points": "5904", "preset": "fine"},
        "warning.frequencies_off_stationary": {"max_force": "6.1e-02", "threshold": "4.5e-04"},
        "warning.frequencies_imaginary": {"values": "-512.3"},
        "warning.optimization_not_converged": {"steps": "64", "max_force": "1.2e-03"},
    }
    assert set(placeholders) == set(WARNING_KEYS), "список ключей разошёлся с тестом"
    for locale in ("ru", "en"):
        for key, params in placeholders.items():
            rendered = t(key, locale, **params)
            assert rendered != key
            assert "{" not in rendered, (locale, key)
            assert rendered.strip()
