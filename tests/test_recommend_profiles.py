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
    """Каждая часть обещания профиля либо выполнена, либо явно снята решением.

    Инвариант сформулирован по правилу, а не по списку нереализованного,
    поэтому он не устареет при появлении D3 или нового функционала.

    Прежняя формулировка («недоступна любая часть обещания — значит HF») была
    строже, чем требует §54 ТЗ: запрет скрывать превратился в запрет
    сообщать. Из-за неё все четыре профиля разворачивались в HF при живых PBE
    и PBE0. Сейчас недоступная дисперсия снимается отдельным решением, а
    недоступный функционал заменяется ближайшим доступным — и то и другое
    обязано быть видно в списке решений.
    """
    registry = default_registry()
    for profile in PrecisionProfile:
        resolution = resolve_profile(profile, task=Task.SINGLE_POINT, molecule=water)
        method = resolution.spec.method
        assert method is not None
        parameters = {decision.parameter for decision in resolution.decisions}

        dispersion = _PROFILE_DISPERSION[profile]
        if dispersion.value != "none" and not registry.is_available(
            f"dispersion:{dispersion.value}"
        ):
            assert method.dispersion.value == "none", profile
            assert "dispersion" in parameters, profile
        else:
            assert method.dispersion is dispersion, profile

        functional = _PROFILE_FUNCTIONAL[profile]
        if registry.is_available(f"functional:{functional}"):
            assert method.functional == functional, profile
            assert method.theory is TheoryFamily.DFT, profile
        else:
            # Запрошенного функционала нет: либо ближайший доступный, либо HF,
            # и в обоих случаях — с явным решением, а не молча.
            assert "functional" in parameters, profile
            if method.theory is TheoryFamily.DFT:
                assert method.functional is not None
                assert registry.is_available(f"functional:{method.functional}")


def test_substitution_prefers_dft_over_hartree_fock(water: Molecule) -> None:
    """Недоступный функционал не утаскивает профиль сразу на HF.

    PBE0 без дисперсии точнее HF всегда, а не только там, где дисперсия не
    важна, поэтому откат на HF допустим лишь тогда, когда реализованных
    функционалов не осталось вовсе. «Исследовательский» профиль просит
    ωB97X-D, которого в ядре нет, — и обязан получить ближайший гибрид.
    """
    registry = default_registry()
    resolution = resolve_profile(PrecisionProfile.RESEARCH, task=Task.SINGLE_POINT, molecule=water)
    method = resolution.spec.method
    assert method is not None
    assert not registry.is_available(f"functional:{_PROFILE_FUNCTIONAL[PrecisionProfile.RESEARCH]}")
    assert method.theory is TheoryFamily.DFT
    assert method.functional is not None
    assert registry.is_available(f"functional:{method.functional}")
    substituted = next(
        decision
        for decision in resolution.decisions
        if decision.parameter == "functional" and decision.reason_key.endswith("substituted")
    )
    assert substituted.value == method.functional
    assert "не реализован" in substituted.render("ru")


def test_hf_fallback_still_happens_without_any_functional(water: Molecule) -> None:
    """Откат на HF не исчез: он срабатывает, когда реализованных DFT нет вовсе.

    Проверка идёт на искусственном реестре без ``method:dft`` и без
    функционалов — иначе путь отката оказался бы непокрытым ровно в тот
    момент, когда он снова понадобится.
    """
    from quantumlab.engine.capabilities import CapabilityKind
    from quantumlab.engine.registry import CapabilityRegistry

    registry = CapabilityRegistry(
        capability
        for capability in default_registry().list_capabilities()
        if capability.kind is not CapabilityKind.FUNCTIONAL and capability.id != "method:dft"
    )
    assert not registry.is_available("method:dft")
    resolution = resolve_profile(
        PrecisionProfile.STANDARD, task=Task.SINGLE_POINT, molecule=water, registry=registry
    )
    method = resolution.spec.method
    assert method is not None
    assert method.theory is TheoryFamily.HF
    assert method.functional is None
    assert method.dispersion.value == "none"
    assert any("не реализован" in line for line in resolution.explain("ru"))
    assert any(
        decision.parameter == "functional" and decision.value == "hf"
        for decision in resolution.decisions
    )


def test_functional_substitution_chain_cannot_loop() -> None:
    """Цепочка замен односторонняя: зацикливание дало бы бесконечный подбор."""
    from quantumlab.recommend.profiles import _FUNCTIONAL_SUBSTITUTES

    for start in _FUNCTIONAL_SUBSTITUTES:
        seen: set[str] = set()
        current = start
        while current:
            assert current not in seen, f"цикл в цепочке замен: {start} -> {current}"
            seen.add(current)
            current = _FUNCTIONAL_SUBSTITUTES.get(current, "")


def test_available_dispersion_is_kept_and_shown(water: Molecule) -> None:
    """Доступная дисперсия профиля остаётся в плане и видна пользователю.

    Профиль «Стандартный расчёт» обещает PBE0-D3(BJ); D3(BJ) реализован,
    поэтому поправка не снимается, а попадает в спецификацию — и строка
    «Дисперсионная поправка: d3bj» присутствует в обоснованиях (§8 ТЗ).
    """
    from quantumlab.domain.spec import DispersionCorrection

    resolution = resolve_profile(PrecisionProfile.STANDARD, task=Task.SINGLE_POINT, molecule=water)
    lines = resolution.explain("ru")
    assert resolution.spec.method is not None
    assert resolution.spec.method.dispersion is DispersionCorrection.D3_BJ
    assert "Дисперсионная поправка: d3bj" in lines
    # Доступная поправка не сопровождается объяснением её «снятия».
    assert not any("не реализована" in line for line in lines)


def test_unavailable_dispersion_is_explained_not_silent(water: Molecule) -> None:
    """Снятая дисперсия видна в обоснованиях вместе с тем, что это значит.

    Правило управляется реестром: если в ядре нет D3(BJ), профиль
    «Стандартный расчёт» обещает PBE0-D3(BJ), и расчёт идёт без поправки;
    пользователь обязан увидеть и факт, и следствие, иначе «стандартный
    расчёт» означал бы не то, что написано (§8 ТЗ). Здесь реестр имитирует
    ядро без D3, чтобы проверить именно ветку снятия.
    """
    from dataclasses import replace

    from quantumlab.engine.capabilities import Availability

    registry = default_registry()
    capability = registry.get("dispersion:d3bj")
    registry.register(
        replace(capability, availability=Availability.NOT_IMPLEMENTED, since_version=None),
        replace=True,
    )
    resolution = resolve_profile(
        PrecisionProfile.STANDARD,
        task=Task.SINGLE_POINT,
        molecule=water,
        registry=registry,
    )
    lines = resolution.explain("ru")
    assert any("Дисперсионная поправка" in line and "не реализована" in line for line in lines)
    assert resolution.spec.method is not None
    assert resolution.spec.method.dispersion.value == "none"
    # Объяснение не дублируется строкой «Дисперсионная поправка: none».
    assert sum("Дисперсионная поправка" in line for line in lines) == 1


def test_high_accuracy_matches_documented_example(water: Molecule) -> None:
    """Пример из ТЗ: «Выбран PBE0/def2-TZVP, потому что профиль Высокая точность».

    PBE0 реализован, поэтому пример проверяется буквально: функционал и базис
    те, что обещаны профилем, а снятая дисперсия объяснена отдельно.
    """
    resolution = resolve_profile(
        PrecisionProfile.HIGH_ACCURACY, task=Task.OPTIMIZATION, molecule=water
    )
    lines = resolution.explain("ru")
    assert "Высокая точность" in lines[0]
    assert "Выбран базисный набор def2-tzvp" in lines
    method = resolution.spec.method
    assert method is not None
    registry = default_registry()
    functional, _, _, dispersion = _base_choice(PrecisionProfile.HIGH_ACCURACY)
    assert registry.is_available(f"functional:{functional}")
    assert f"Выбран функционал {functional}" in lines
    assert method.theory is TheoryFamily.DFT
    # D3(BJ) в ядре нет: расчёт идёт без поправки, и это объяснено, а не спрятано.
    if not registry.is_available(f"dispersion:{dispersion.value}"):
        assert method.dispersion.value == "none"
        assert any("не реализована" in line and "Дисперсионная" in line for line in lines)


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


def test_research_profile_tightens_numerics_and_damps(water: Molecule) -> None:
    """Исследовательский профиль ужесточает численные параметры.

    Проверка устойчивости сюда больше не входит безусловно: ядро её не
    реализует, и включить её в спецификацию значило бы сделать профиль
    незапускаемым. Ожидание читается из реестра, а не зашито: когда возможность
    появится, тест сам начнёт требовать ``True``.
    """
    resolution = resolve_profile(PrecisionProfile.RESEARCH, task=Task.SINGLE_POINT, molecule=water)
    available = default_registry().is_available("scf:stability_analysis")
    assert resolution.spec.scf.stability_analysis is available
    assert resolution.spec.scf.damping > 0.0
    assert resolution.spec.scf.energy_threshold <= 1e-10


def test_every_profile_produces_a_spec_the_engine_accepts(water: Molecule) -> None:
    """Подобранные параметры обязаны быть работоспособными.

    Инвариант, а не проверка одного случая: подборщик и движок развиваются
    независимо, и расхождение между ними означает, что «подобранные параметры»
    отклоняются при запуске. Пользователь нажимает «Рассчитать» после выбора
    точности (§17 ТЗ) и не должен получать отказ от того, что система сама же
    и подобрала.

    Регрессия, которую этот тест ловит, уже случалась: ядро начало отклонять
    ``stability_analysis``, а точные профили его запрашивали — и два профиля из
    четырёх перестали запускаться.
    """
    from quantumlab.engine.reference import ReferenceEngine

    engine = ReferenceEngine()
    supported = {Task(name) for name in engine.supported_tasks()}

    checked = 0
    for profile in PrecisionProfile:
        for task in supported:
            resolution = resolve_profile(profile, task=task, molecule=water)
            assert engine.assert_supported(resolution.spec), (profile, task)
            checked += 1
    assert checked == len(list(PrecisionProfile)) * len(supported)


def test_unavailable_scf_option_is_reported_as_a_decision(water: Molecule) -> None:
    """Пропуск нереализованной опции объясняется, а не происходит молча.

    Если профиль обещает проверку устойчивости, а ядро её не делает,
    пользователь обязан это увидеть: иначе «высокая точность» означала бы
    нечто иное, чем написано (§8 ТЗ).
    """
    from quantumlab.engine.registry import default_registry

    registry = default_registry()
    if registry.is_available("scf:stability_analysis"):
        pytest.skip("проверка устойчивости реализована — пропускать нечего")
    resolution = resolve_profile(
        PrecisionProfile.HIGH_ACCURACY, task=Task.SINGLE_POINT, molecule=water
    )
    assert resolution.spec.scf.stability_analysis is False
    assert any(decision.parameter == "stability_analysis" for decision in resolution.decisions)
    rendered = next(
        decision.render("ru")
        for decision in resolution.decisions
        if decision.parameter == "stability_analysis"
    )
    assert "не реализована" in rendered
