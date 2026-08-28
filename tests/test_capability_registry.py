"""Реестр возможностей: честность статусов (§54 ТЗ) и расширение плагинами."""

from __future__ import annotations

import pytest

from quantumlab.engine.capabilities import Availability, Capability, CapabilityKind
from quantumlab.engine.registry import CapabilityRegistry, default_registry
from quantumlab.errors import BasisNotFoundError, FunctionalNotFoundError, MethodNotAvailableError


@pytest.fixture(scope="module")
def registry() -> CapabilityRegistry:
    return default_registry()


#: Все базисы, которые реестр обязан знать. Используется несколькими тестами,
#: чтобы список не разъехался между ними.
_BASIS_NAMES = {
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g(d)",
    "6-31g(d,p)",
    "6-311g",
    "6-311g(d,p)",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
}


def test_all_declared_basis_sets_are_present(registry: CapabilityRegistry) -> None:
    declared = {item.name for item in registry.list_capabilities(CapabilityKind.BASIS)}
    assert declared >= _BASIS_NAMES


def test_all_declared_methods_and_functionals_are_present(registry: CapabilityRegistry) -> None:
    methods = {item.name for item in registry.list_capabilities(CapabilityKind.METHOD)}
    assert {"hf", "dft", "mp2", "scs_mp2", "ccsd", "ccsd_t"} <= methods
    functionals = {item.name for item in registry.list_capabilities(CapabilityKind.FUNCTIONAL)}
    assert {
        "pbe",
        "blyp",
        "pbe0",
        "b3lyp",
        "tpssh",
        "m06",
        "m062x",
        "wb97x",
        "wb97x-d",
    } <= functionals


def test_unimplemented_methods_are_reported_honestly(registry: CapabilityRegistry) -> None:
    """Все методы, кроме HF, остаются недоступными: кода для них нет.

    HF — единственное исключение, и он помечен ``partial``, а не
    ``implemented``: реализован только RHF и только для двух задач.
    """
    hf = registry.get("method:hf")
    assert hf.availability is Availability.PARTIAL
    assert hf.is_usable
    # Ограничения перечислены явно: по спину, по набору задач и по координатам.
    assert len(hf.limitations) >= 3
    assert any("RHF" in text for text in hf.limitations)

    methods = [
        item
        for item in registry.list_capabilities(CapabilityKind.METHOD)
        if item.id.startswith("method:")
    ]
    assert {item.id for item in methods if item.is_usable} == {"method:hf", "method:dft"}
    for capability in methods:
        if capability.id in ("method:hf", "method:dft"):
            continue
        assert capability.availability is Availability.NOT_IMPLEMENTED, capability.id
        assert not capability.is_usable, capability.id

    # DFT работает, но только в варианте LDA-функционала SVWN и только для
    # энергии в точке: «partial» без перечня ограничений выглядел бы как «готово».
    dft = registry.get("method:dft")
    assert dft.availability is Availability.PARTIAL
    assert dft.limitations
    assert any("SVWN" in text for text in dft.limitations)
    assert any("градиент" in text for text in dft.limitations)
    # Реализованный функционал виден в реестре, заявленный без кода — нет.
    assert registry.is_available("functional:svwn")
    assert registry.is_available("functional:pbe")
    assert registry.is_available("functional:pbe0")
    assert not registry.is_available("functional:b3lyp")
    assert not registry.is_available("method:ccsd_t")
    assert not registry.is_available("spin:rohf")

    # UHF реализован, но не целиком: ограничения обязаны быть перечислены,
    # иначе «partial» выглядело бы как «готово» (§54 ТЗ). Аналитические
    # градиенты UHF появились, поэтому про них здесь больше не утверждаем —
    # проверяются те ограничения, что действительно остались.
    uhf = registry.get("spin:uhf")
    assert uhf.availability is Availability.PARTIAL
    assert uhf.is_usable
    assert uhf.limitations
    assert any("декартов" in text for text in uhf.limitations)
    assert any("S^2" in text for text in uhf.limitations)


def test_assert_available_raises_localized_error(registry: CapabilityRegistry) -> None:
    with pytest.raises(MethodNotAvailableError) as info:
        registry.assert_available("method:mp2")
    assert info.value.title("ru") == "Метод пока недоступен"
    assert "MP2" in info.value.explain("en") or "mp2" in info.value.explain("en")


def test_assert_available_distinguishes_error_types(registry: CapabilityRegistry) -> None:
    # Все 16 зарегистрированных базисов теперь пригодны к расчёту, поэтому
    # BasisNotFoundError даёт только имя, которого в реестре нет.
    with pytest.raises(BasisNotFoundError):
        registry.assert_available("basis:unobtainium-qzvp")
    with pytest.raises(FunctionalNotFoundError):
        registry.assert_available("functional:b3lyp")
    # optimization реализована, поэтому примером недоступной задачи служат
    # частоты: для них нужен гессиан, которого нет.
    with pytest.raises(MethodNotAvailableError):
        registry.assert_available("task:frequencies")


def test_unknown_capability_is_treated_as_unavailable(registry: CapabilityRegistry) -> None:
    assert registry.availability("method:does-not-exist") is Availability.NOT_IMPLEMENTED
    assert registry.find("nope") is None


def test_lookup_is_case_and_space_insensitive(registry: CapabilityRegistry) -> None:
    assert registry.find("6-31G(D,P)") is not None
    found = registry.find("  def2-TZVP ")
    assert found is not None
    assert found.id == "basis:def2-tzvp"


def test_implemented_capabilities_match_the_verified_surface(
    registry: CapabilityRegistry,
) -> None:
    """``implemented`` ровно там, где есть проверенный код.

    Список сверен с фактическим состоянием: XYZ-парсер, single_point,
    reference-cpu и шесть базисов, опубликованных в декартовой схеме.
    """
    implemented = sorted(
        item.id
        for item in registry.list_capabilities(available_only=True)
        if item.availability is Availability.IMPLEMENTED
    )
    assert implemented == [
        "backend:reference-cpu",
        "basis:3-21g",
        "basis:6-311g",
        "basis:6-31g",
        "basis:6-31g(d)",
        "basis:6-31g(d,p)",
        "basis:sto-3g",
        # Единственная доступная «дисперсионная поправка» — её отсутствие.
        # D3/D4 в реестре есть и помечены как нереализованные.
        "dispersion:none",
        "format:xyz",
        "task:optimization",
        "task:single_point",
    ]
    # Оптимизация реализована, но только в декартовых координатах — поэтому
    # coordinates:cartesian partial, а внутренние координаты не реализованы.
    assert registry.get("coordinates:cartesian").availability is Availability.PARTIAL
    assert not registry.is_available("coordinates:redundant_internal")
    assert not registry.is_available("task:frequencies")
    assert registry.get("format:xyz").describe("ru") == (
        "Референсная реализация: проверена на верификационном наборе."
    )


def test_basis_availability_follows_the_published_angular_scheme(
    registry: CapabilityRegistry,
) -> None:
    """Статус базиса читается из данных, а не хранится вторым мнением.

    Реестр не должен расходиться с тем, что записал генератор из Basis Set
    Exchange: сферическая публикация d/f означает, что наш декартов расчёт
    даёт больший базис, и это ограничение обязано быть видимым.
    """
    cartesian = {"sto-3g", "3-21g", "6-31g", "6-31g(d)", "6-31g(d,p)", "6-311g"}
    for name in _BASIS_NAMES:
        capability = registry.get(f"basis:{name}")
        scheme = capability.metadata["angular_scheme_published"]
        if name in cartesian:
            assert scheme == "cartesian", name
            assert capability.availability is Availability.IMPLEMENTED, name
            assert not capability.limitations, name
        else:
            assert scheme == "spherical", name
            assert capability.availability is Availability.PARTIAL, name
            assert "сферической" in capability.limitations[0], name
        # Обе схемы пригодны к расчёту — отличается только точность сравнения
        # с табличными значениями.
        assert capability.is_usable, name


def test_snapshot_groups_by_kind(registry: CapabilityRegistry) -> None:
    snapshot = registry.snapshot()
    assert set(snapshot) == {
        "task",
        "method",
        "functional",
        "dispersion",
        "basis",
        "format",
        "backend",
        "scheduler",
        "coordinates",
        "spin",
    }
    assert snapshot["backend"][0]["availability"] == "not_implemented"
    # Системы координат и спин — отдельные категории: раздел «База методов» в GUI
    # строится из этого среза, и координаты в списке методов были бы ошибкой.
    assert all(str(item["id"]).startswith("coordinates:") for item in snapshot["coordinates"])
    assert all(str(item["id"]).startswith("spin:") for item in snapshot["spin"])
    assert not any(
        str(item["id"]).startswith(("coordinates:", "spin:")) for item in snapshot["method"]
    )


def test_plugin_can_register_new_capability(registry: CapabilityRegistry) -> None:
    plugin_registry = CapabilityRegistry(registry.list_capabilities())
    plugin_registry.register(
        Capability(
            id="method:gfnn-x_tb",
            kind=CapabilityKind.METHOD,
            name="gfnn-xtb",
            availability=Availability.PARTIAL,
            since_version="0.2.0",
            limitations=("только органические молекулы",),
            notes_key="capability.note.partial",
        )
    )
    capability = plugin_registry.assert_available("method:gfnn-x_tb")
    assert capability.availability is Availability.PARTIAL
    assert "только органические молекулы" in capability.describe("ru")


def test_duplicate_registration_is_rejected(registry: CapabilityRegistry) -> None:
    with pytest.raises(ValueError, match="уже зарегистрирована"):
        registry.register(registry.get("method:hf"))
