"""Автоподбор параметров (§8 ТЗ): правильность выбора и объяснимость решений."""

from __future__ import annotations

from pathlib import Path

import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import GridPreset, PrecisionProfile, Task, TheoryFamily
from quantumlab.engine.registry import default_registry
from quantumlab.recommend.profiles import (
    HardwareContext,
    _base_choice,
    resolve_profile,
)

#: Какой функционал обещает каждый профиль. Берётся из самой таблицы выбора,
#: а не переписывается здесь: вторая копия разошлась бы с первой при первом же
#: изменении профилей, и тест начал бы проверять не то.
_PROFILE_FUNCTIONAL = {profile: _base_choice(profile)[0] for profile in PrecisionProfile}
_PROFILE_DISPERSION = {profile: _base_choice(profile)[3] for profile in PrecisionProfile}

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


@pytest.fixture
def big_system() -> Molecule:
    """Цепочка из 100 атомов углерода — модель крупной системы."""
    return Molecule.from_atoms(
        ["C"] * 100,
        [(1.5 * index, 0.0, 0.0) for index in range(100)],
        name="chain-100",
    )


@pytest.mark.parametrize(
    ("profile", "basis"),
    [
        (PrecisionProfile.SCREENING, "def2-svp"),
        (PrecisionProfile.STANDARD, "def2-svp"),
        (PrecisionProfile.HIGH_ACCURACY, "def2-tzvp"),
        (PrecisionProfile.RESEARCH, "def2-tzvp"),
    ],
)
def test_profile_selects_documented_basis(
    water: Molecule, profile: PrecisionProfile, basis: str
) -> None:
    resolution = resolve_profile(profile, task=Task.SINGLE_POINT, molecule=water)
    method = resolution.spec.method
    assert method is not None
    assert method.basis == basis
    assert resolution.spec.profile is profile


def test_plan_only_recommends_what_the_engine_can_run(water: Molecule) -> None:
    """§54 ТЗ: план не имеет права предлагать то, что не может выполниться.

    Инвариант сформулирован без предположений о том, что именно сейчас
    реализовано: **каждый** рекомендованный метод и функционал обязаны быть
    доступны в реестре. План с нереализованным функционалом — это расчёт,
    который гарантированно упадёт уже после нажатия «Рассчитать».

    Пока ни один из функционалов профилей (PBE, PBE0, ωB97X-D) не реализован,
    все профили откатываются на HF; когда появятся, тест продолжит проверять
    тот же инвариант, а не устареет.
    """
    registry = default_registry()
    for profile in PrecisionProfile:
        method = resolve_profile(profile, task=Task.SINGLE_POINT, molecule=water).spec.method
        assert method is not None
        assert registry.is_available(f"method:{method.theory.value}")
        if method.functional is not None:
            assert method.theory is TheoryFamily.DFT
            assert registry.is_available(f"functional:{method.functional}"), method.functional
        else:
            assert method.theory is TheoryFamily.HF
            assert method.dispersion.value == "none"


def test_profiles_never_promise_what_the_engine_cannot_do(water: Molecule) -> None:
    """Профиль разворачивается в HF, если недоступна любая часть его обещания.

    Инвариант сформулирован по правилу, а не по списку нереализованного,
    поэтому он не устареет при появлении D3 или нового функционала.

    Сейчас у «Скрининга» реализован PBE, но нет D3(BJ), а у
    «Исследовательского» нет ωB97X-D — и тот и другой обязаны откатиться.
    Учитывать только функционал было бы ошибкой: план пообещал бы
    дисперсионную поправку, которой не существует, и расчёт выполнился бы как
    другой (§54 ТЗ).
    """
    registry = default_registry()
    for profile in PrecisionProfile:
        functional_ok = registry.is_available(f"functional:{_PROFILE_FUNCTIONAL[profile]}")
        dispersion_ok = registry.is_available(f"dispersion:{_PROFILE_DISPERSION[profile].value}")
        method = resolve_profile(profile, task=Task.SINGLE_POINT, molecule=water).spec.method
        assert method is not None
        if functional_ok and dispersion_ok:
            assert method.theory is TheoryFamily.DFT, profile
        else:
            assert method.theory is TheoryFamily.HF, profile


def test_hf_fallback_is_explained_not_silent(water: Molecule) -> None:
    """Откат виден в обоснованиях: профиль обещает точность DFT, а даёт HF."""
    resolution = resolve_profile(PrecisionProfile.STANDARD, task=Task.SINGLE_POINT, molecule=water)
    assert any("не реализован" in line for line in resolution.explain("ru"))
    assert any(
        decision.parameter == "functional" and decision.value == "hf"
        for decision in resolution.decisions
    )


def test_high_accuracy_matches_documented_example(water: Molecule) -> None:
    """Пример из ТЗ: «Выбран PBE0/def2-TZVP, потому что профиль Высокая точность».

    Пример станет буквально верен, когда появится DFT. До тех пор проверяется
    честное поведение: базис тот же, а вместо недоступного функционала — HF с
    явным объяснением, а не молчаливая подмена.
    """
    resolution = resolve_profile(
        PrecisionProfile.HIGH_ACCURACY, task=Task.OPTIMIZATION, molecule=water
    )
    lines = resolution.explain("ru")
    assert "Высокая точность" in lines[0]
    assert "Выбран базисный набор def2-tzvp" in lines
    method = resolution.spec.method
    assert method is not None
    # Условие привязано к доступности самого функционала, а не к семейству
    # методов: DFT может быть реализован частично, и тогда обещать PBE0 нельзя.
    if default_registry().is_available("functional:pbe0"):
        assert "Выбран функционал pbe0" in lines
    else:
        assert method.theory is TheoryFamily.HF
        assert any("не реализован" in line for line in lines)


def test_frequencies_tighten_numerics(water: Molecule) -> None:
    single_point = resolve_profile(
        PrecisionProfile.STANDARD, task=Task.SINGLE_POINT, molecule=water
    )
    frequencies = resolve_profile(PrecisionProfile.STANDARD, task=Task.FREQUENCIES, molecule=water)
    assert frequencies.spec.scf.energy_threshold <= 1e-10
    assert frequencies.spec.scf.energy_threshold < single_point.spec.scf.energy_threshold
    assert frequencies.spec.grid.preset is GridPreset.ULTRAFINE


def test_transition_state_uses_smaller_trust_radius(water: Molecule) -> None:
    optimization = resolve_profile(
        PrecisionProfile.STANDARD, task=Task.OPTIMIZATION, molecule=water
    )
    transition_state = resolve_profile(
        PrecisionProfile.STANDARD, task=Task.TS_OPTIMIZATION, molecule=water
    )
    assert (
        transition_state.spec.optimization.trust_radius
        < optimization.spec.optimization.trust_radius
    )
    assert transition_state.spec.optimization.hessian_update == "bofill"


def test_large_system_trades_accuracy_for_time(big_system: Molecule) -> None:
    resolution = resolve_profile(
        PrecisionProfile.HIGH_ACCURACY, task=Task.OPTIMIZATION, molecule=big_system
    )
    assert resolution.spec.method is not None
    assert resolution.spec.method.basis == "def2-svp"
    assert resolution.spec.grid.preset is GridPreset.FINE
    basis_decision = next(d for d in resolution.decisions if d.parameter == "basis")
    assert basis_decision.detail is not None and "100 атомов" in basis_decision.detail


def test_resources_never_exceed_available_hardware(water: Molecule) -> None:
    hardware = HardwareContext(cores=4, memory_mb=2048, gpu_count=0)
    resolution = resolve_profile(
        PrecisionProfile.STANDARD, task=Task.OPTIMIZATION, molecule=water, hardware=hardware
    )
    resources = resolution.spec.resources
    assert resources.threads is not None and 1 <= resources.threads <= 4
    assert resources.memory_mb is not None and resources.memory_mb <= 2048


def test_every_decision_is_localized_in_both_languages(water: Molecule) -> None:
    resolution = resolve_profile(PrecisionProfile.STANDARD, task=Task.OPTIMIZATION, molecule=water)
    assert len(resolution.decisions) >= 7
    for locale in ("ru", "en"):
        rendered = resolution.explain(locale)
        assert len(rendered) == len(resolution.decisions) + 1
        assert all(line and "{" not in line for line in rendered)


def test_research_profile_enables_stability_analysis_and_damping(water: Molecule) -> None:
    resolution = resolve_profile(PrecisionProfile.RESEARCH, task=Task.SINGLE_POINT, molecule=water)
    assert resolution.spec.scf.stability_analysis is True
    assert resolution.spec.scf.damping > 0.0
    assert resolution.spec.scf.energy_threshold <= 1e-10
