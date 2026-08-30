"""Сверка DFT-D3 с независимой сборкой s-dftd3 (pyscf-dispersion).

По ADR-002 PySCF используется **только как оракул**: our engine never reads
its tables or calls its routines in production. The bundled s-dftd3 library
inside ``pyscf-dispersion`` is an independently built, battle-tested
implementation of the same model (s-dftd3 v1.2.1 line), so agreement with it
is a real verification, not a self-check.

Погрешность. Константа Å→bohr в s-dftd3 (``AATOA = 1.8897261246``) отличается
от нашей (``1.8897259885789``) на ~7e-8 по относительной величине. Поэтому
сопоставляем с оракулом с допуском, заведомо больше этой разности:
относительное 1e-5 по энергии/силам. Тесты, не зависящие от преобразования
в единиц (FD-градиент, трансляционная инвариантность), живут в
``test_dispersion.py`` с допуском 1e-10.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.spec import DispersionCorrection
from quantumlab.engine import dispersion as d3

pyscf = pytest.importorskip("pyscf", reason="pyscf-dispersion нужен только для сверки")
#: Подмодуль регистрируется через importlib: статический импорт заставил бы
#: mypy искать у PySCF заглушки типов, которых нет.
importlib.import_module("pyscf.dispersion.dftd3")

pytestmark = pytest.mark.scientific

FIXTURES = Path(__file__).parent / "fixtures"

#: Допуск сверки с оракулом: больше разницы в константе Å→bohr (7e-8)
#: на несколько порядков, но всё ещё на пять порядков меньше самой величины.
REL_TOL: float = 1e-5


def _water() -> Molecule:
    return Molecule.from_xyz((FIXTURES / "water.xyz").read_text(encoding="utf-8"), name="water")


def _h2(distance_angstrom: float) -> Molecule:
    return Molecule(
        atoms=(
            _atom("H", (0.0, 0.0, 0.0)),
            _atom("H", (distance_angstrom, 0.0, 0.0)),
        )
    )


def _atom(symbol: str, position: tuple[float, float, float]) -> Atom:
    return Atom(symbol=symbol, position=position)


def _oracle(molecule: Molecule, xc: str, version: str) -> tuple[float, np.ndarray]:
    """Энергия и градиент D3 из независимой сборки s-dftd3 (э, э/bohr)."""
    mol = pyscf.gto.M(
        atom=[(atom.symbol, list(atom.position)) for atom in molecule.atoms],
        unit="Angstrom",
        verbose=0,
    )
    disp = pyscf.dispersion.dftd3.DFTD3Dispersion(mol, xc=xc, version=version, atm=False)
    result = disp.get_dispersion(grad=True)
    return float(result["energy"]), np.asarray(result["gradient"])


@pytest.mark.parametrize(
    ("correction", "xc"),
    [
        (DispersionCorrection.D3_BJ, "hf"),
        (DispersionCorrection.D3_BJ, "pbe"),
        (DispersionCorrection.D3_BJ, "pbe0"),
        (DispersionCorrection.D3_BJ, "blyp"),
        (DispersionCorrection.D3_BJ, "b3lyp"),
        (DispersionCorrection.D3_ZERO, "hf"),
        (DispersionCorrection.D3_ZERO, "pbe"),
        (DispersionCorrection.D3_ZERO, "pbe0"),
        (DispersionCorrection.D3_ZERO, "blyp"),
        (DispersionCorrection.D3_ZERO, "b3lyp"),
    ],
)
def test_water_matches_oracle(correction: DispersionCorrection, xc: str) -> None:
    molecule = _water()
    ours = d3.dftd3_contribution(molecule, correction, None if xc == "hf" else xc)
    ref_energy, ref_gradient = _oracle(molecule, xc, correction.value)

    assert abs(ours.energy_hartree - ref_energy) <= REL_TOL * abs(ref_energy)
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


@pytest.mark.parametrize("distance", (0.74, 1.5, 2.5, 4.0))
def test_h2_bj_hf_matches_oracle_over_distance(distance: float) -> None:
    molecule = _h2(distance)
    ours = d3.dftd3_contribution(molecule, DispersionCorrection.D3_BJ, None)
    ref_energy, ref_gradient = _oracle(molecule, "hf", "d3bj")
    assert abs(ours.energy_hartree - ref_energy) <= REL_TOL * abs(ref_energy)
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


def test_gradient_unit_is_hartree_per_bohr() -> None:
    # Оракул возвращает градиент в э/bohr при координатах в Å. Если бы наш
    # модуль выдавал э/Å, расхождение было бы ровно в 1.8897 раза — на много
    # больше допуска REL_TOL.
    molecule = _water()
    ours = d3.dftd3_contribution(molecule, DispersionCorrection.D3_BJ, "pbe")
    _, ref_gradient = _oracle(molecule, "pbe", "d3bj")
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


def test_engine_total_energy_includes_d3() -> None:
    # Полная энергия из движка = электронная (SCF) + D3. Сверяем с PySCF,
    # который складывает то же самое: RHF/PBE SCF + D3 из независимой сборки.
    from quantumlab.domain.spec import CalculationSpec, MethodSpec, Task, TheoryFamily
    from quantumlab.engine.contracts import EngineRequest
    from quantumlab.engine.reference import ReferenceEngine

    molecule = _water()
    for theory, functional, xc in [
        (TheoryFamily.HF, None, "hf"),
        (TheoryFamily.DFT, "pbe", "pbe"),
    ]:
        spec = CalculationSpec(
            task=Task.SINGLE_POINT,
            method=MethodSpec(
                theory=theory,
                functional=functional,
                basis="sto-3g",
                dispersion=DispersionCorrection.D3_BJ,
            ),
        )
        result = ReferenceEngine().run(
            EngineRequest(job_id="d3-xcheck", spec=spec, molecule=molecule, threads=1)
        )
        mol = pyscf.gto.M(
            atom=[(a.symbol, list(a.position)) for a in molecule.atoms],
            unit="Angstrom",
            verbose=0,
        )
        scf_energy = (
            mol.RHF().run().e_tot if theory is TheoryFamily.HF else mol.RKS(xc=xc).run().e_tot
        )
        d3_energy = pyscf.dispersion.dftd3.DFTD3Dispersion(
            mol, xc=xc, version="d3bj", atm=False
        ).get_dispersion()["energy"]
        assert result.dispersion_energy_hartree is not None
        # Вклад D3 — главный предмет сверки: он обязан совпасть с
        # независимой сборкой в пределах REL_TOL.
        assert abs(result.dispersion_energy_hartree - d3_energy) <= REL_TOL * abs(d3_energy)
        # Полная сумма: электронный вклад — это разность двух независимых
        # SCF (наш движок и PySCF, разные сетки интеграции для DFT), и её
        # расхождение (до ~1e-6 для PBE) не относится к D3; поэтому на
        # полную сумму даём отдельный, более крупный допуск и проверяем
        # электронный и дисперсионный вклады раздельно.
        our_scf = result.energy_hartree - result.dispersion_energy_hartree
        assert abs(our_scf - scf_energy) <= 2e-6
        assert abs(result.energy_hartree - (scf_energy + d3_energy)) <= 2e-6
