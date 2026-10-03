"""Пользовательские базисы: разбор Gaussian94, каталог, реестр, CLI и расчёт."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantumlab.cli import main
from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import available_basis_sets, build_basis
from quantumlab.engine.basis_custom import (
    CustomBasisError,
    basis_directory,
    import_basis_file,
    parse_gaussian94,
)
from quantumlab.engine.registry import default_registry
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.errors import BasisNotFoundError

#: STO-3G для H и O в формате Gaussian94 — значения из публикации с 8 знаками,
#: поэтому энергия совпадает со встроенным ``sto-3g`` до 1e-6 (округление чисел).
STO3G_GBS = """\
! STO-3G (копия для теста)
****
H     0
S   3   1.00
      3.42525091             0.15432897
      0.62391373             0.53532814
      0.16885540             0.44463454
****
O     0
S   3   1.00
    130.7093200              0.15432897
     23.8088610              0.53532814
      6.4436083              0.44463454
SP   3   1.00
      5.0331513             -0.09996723             0.15591627
      1.1695961              0.39951283             0.60768372
      0.3803890              0.70011547             0.39195739
****
"""

FIXTURES = Path(__file__).parent / "fixtures"


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def test_parse_gaussian94_matches_the_bse_schema() -> None:
    raw = parse_gaussian94(STO3G_GBS, name="mine")
    assert raw["name"] == "mine"
    oxygen = raw["elements"]["8"]["shells"]
    assert [shell["angular_momentum"] for shell in oxygen] == [[0], [0, 1]]
    assert oxygen[1]["coefficients"][1][0] == pytest.approx(0.15591627)
    assert raw["elements"]["1"]["shells"][0]["exponents"][0] == pytest.approx(3.42525091)


def test_fortran_exponents_are_accepted() -> None:
    text = "****\nH 0\nS 1 1.00\n 1.0D+00 1.0D+00\n****\n"
    raw = parse_gaussian94(text, name="d")
    assert raw["elements"]["1"]["shells"][0]["exponents"] == [1.0]


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "ни одного элемента"),
        ("****\nXx 0\n", "заголовок элемента"),
        ("****\nH 0\nS 2 1.00\n 1.0 1.0\n", "оборвана"),
        ("****\nH 0\nS 1 1.00\n 1.0\n****\n", "ожидалось 2 чисел"),
        ("****\nH 0\nS 1 1.00\n -1.0 1.0\n****\n", "положительными"),
        ("****\nH 0\nS 0 1.00\n****\n", "без примитивов"),
        ("****\nH 0\nS 1 1.00\n abc 1.0\n****\n", "не число"),
        ("****\nH 0\n****\n", "нет ни одной оболочки"),
    ],
)
def test_malformed_files_are_rejected_with_a_reason(text: str, fragment: str) -> None:
    with pytest.raises(CustomBasisError, match=fragment):
        parse_gaussian94(text, name="bad")


def test_custom_basis_gives_the_same_energy_as_the_builtin_copy(tmp_path: Path) -> None:
    source = tmp_path / "MySTO.gbs"
    source.write_text(STO3G_GBS, encoding="utf-8")
    target = import_basis_file(source, scheme="cartesian")
    assert target.parent == basis_directory()
    assert target.stem == "mysto"
    assert "mysto" in available_basis_sets()

    water = _water()
    settings = ScfSettings(energy_tolerance=1e-10, density_tolerance=1e-8)
    custom = run_rhf(build_basis("mysto", water), water, settings)
    builtin = run_rhf(build_basis("sto-3g", water), water, settings)
    assert custom.converged
    assert custom.total_energy == pytest.approx(builtin.total_energy, abs=1e-6)


def test_custom_basis_display_name_carries_a_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "mine.gbs"
    source.write_text(STO3G_GBS, encoding="utf-8")
    import_basis_file(source, scheme="cartesian")
    first = build_basis("mine", _water()).display_name
    source.write_text(STO3G_GBS.replace("0.62391373", "0.62391374"), encoding="utf-8")
    import_basis_file(source, scheme="cartesian")
    second = build_basis("mine", _water()).display_name
    assert "custom" in first
    assert first != second


def test_builtin_name_cannot_be_shadowed(tmp_path: Path) -> None:
    source = tmp_path / "sto-3g.gbs"
    source.write_text(STO3G_GBS, encoding="utf-8")
    with pytest.raises(CustomBasisError, match="занято"):
        import_basis_file(source, builtin_names=frozenset(available_basis_sets()))


def test_json_import_is_validated(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text(json.dumps({"elements": {"1": {"shells": [{"exponents": [1.0]}]}}}))
    with pytest.raises(CustomBasisError, match="неполная оболочка"):
        import_basis_file(source)


def test_unknown_basis_is_still_unknown() -> None:
    with pytest.raises(BasisNotFoundError):
        build_basis("not-imported", _water())


def test_registry_lists_imported_basis_as_custom(tmp_path: Path) -> None:
    source = tmp_path / "mine.gbs"
    source.write_text(STO3G_GBS, encoding="utf-8")
    import_basis_file(source, scheme="cartesian")
    capability = default_registry().get("basis:mine")
    assert capability.is_usable
    assert capability.metadata["origin"] == "custom"


def test_cli_import_then_plan_uses_the_basis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "mine.gbs"
    source.write_text(STO3G_GBS, encoding="utf-8")
    assert main(["basis", "import", str(source), "--scheme", "cartesian"]) == 0
    assert "mine" in capsys.readouterr().out
    xyz = FIXTURES / "water.xyz"
    code = main(
        [
            "plan",
            str(xyz),
            "--task",
            "energy",
            "--method",
            "hf",
            "--basis",
            "mine",
        ]
    )
    assert code == 0
    assert "mine" in capsys.readouterr().out
