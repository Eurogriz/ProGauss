"""Сверка DFT-D4 с независимой сборкой libdftd4 (pyscf-dispersion).

По ADR-002 PySCF используется **только как оракул**: движок не читает его
таблицы и не вызывает его рутин в production. libdftd4 4.0.1 внутри
pyscf-dispersion — независимая battle-tested сборка той же модели (dftd4
v4.0.1: параметры bj-eeq-atm, EEQ eeq2019), так что совпадение с ней —
реальная верификация, а не самопроверка (таблицы нашей реализации
извлечены из исходников dftd4, но сама сборка — другая кодовая база:
Fortran libdftd4 против нашей чистой Python).

Погрешность. Константа Å→bohr в PySCF (1.889726124564) отличается от нашей
(1.889726133890, ``aatoau`` dftd4, CODATA-2018) на ~5e-9 по относительной
величине. Допуск REL_TOL на порядки больше этой разности, но на порядки
меньше самой величины — как в ``test_crosscheck_dftd3``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Atom, Molecule
from quantumlab.domain.spec import DispersionCorrection
from quantumlab.engine import dispersion as d4mod

pyscf = pytest.importorskip("pyscf", reason="pyscf-dispersion нужен только для сверки")
#: Подмодуль регистрируется через importlib: статический импорт заставил бы
#: mypy искать у PySCF заглушки типов, которых нет.
importlib.import_module("pyscf.dispersion.dftd4")

pytestmark = pytest.mark.scientific

FIXTURES = Path(__file__).parent / "fixtures"

#: Допуск сверки с оракулом: больше разницы в константе Å→bohr (5e-9)
#: на порядки, но всё ещё на порядки меньше самой величины.
REL_TOL: float = 1e-5

#: Функционалы D4 (v4.0.1, 118 обученных) — выборочное покрытие: hf (нет XC),
#: GGA, гибриды, RSH с отрицательным s8.
XCS: tuple[str, ...] = ("hf", "pbe", "pbe0", "blyp", "b3lyp", "tpss", "wb97x-2008")


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


def _oracle(molecule: Molecule, xc: str) -> tuple[float, np.ndarray]:
    """Энергия и градиент D4 из независимой сборки libdftd4 (э, э/bohr)."""
    mol = pyscf.gto.M(
        atom=[(atom.symbol, list(atom.position)) for atom in molecule.atoms],
        unit="Angstrom",
        verbose=0,
    )
    disp = pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc)
    result = disp.get_dispersion(grad=True)
    return float(result["energy"]), np.asarray(result["gradient"])


@pytest.mark.parametrize("xc", XCS)
def test_water_matches_oracle(xc: str) -> None:
    molecule = _water()
    ours = d4mod.dftd4_contribution(molecule, None if xc == "hf" else xc)
    ref_energy, ref_gradient = _oracle(molecule, xc)

    assert abs(ours.energy_hartree - ref_energy) <= REL_TOL * abs(ref_energy)
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


@pytest.mark.parametrize("distance", (0.74, 1.5, 2.5, 4.0))
def test_h2_hf_matches_oracle_over_distance(distance: float) -> None:
    molecule = _h2(distance)
    ours = d4mod.dftd4_contribution(molecule, None)
    ref_energy, ref_gradient = _oracle(molecule, "hf")
    assert abs(ours.energy_hartree - ref_energy) <= REL_TOL * abs(ref_energy)
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


def test_gradient_unit_is_hartree_per_bohr() -> None:
    # libdft4 возвращает градиент в э/bohr при координатах в Å (проверено
    # конечными разностями: FD по Å × 1/AATOA = вернутое значение). Если бы
    # наш модуль выдавал э/Å, расхождение было бы ровно в 1.8897 раза —
    # на много больше допуска REL_TOL.
    molecule = _water()
    ours = d4mod.dftd4_contribution(molecule, "pbe")
    _, ref_gradient = _oracle(molecule, "pbe")
    assert np.max(np.abs(ours.gradient - ref_gradient)) <= REL_TOL * max(
        1e-30, np.max(np.abs(ref_gradient))
    )


def test_engine_total_energy_includes_d4() -> None:
    # Полная энергия из движка = электронная (SCF) + D4. Сверяем с PySCF,
    # который складывает то же самое: SCF + D4 из независимой сборки.
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
                dispersion=DispersionCorrection.D4,
            ),
        )
        result = ReferenceEngine().run(
            EngineRequest(job_id="d4-xcheck", spec=spec, molecule=molecule, threads=1)
        )
        mol = pyscf.gto.M(
            atom=[(a.symbol, list(a.position)) for a in molecule.atoms],
            unit="Angstrom",
            verbose=0,
        )
        scf_energy = (
            mol.RHF().run().e_tot if theory is TheoryFamily.HF else mol.RKS(xc=xc).run().e_tot
        )
        d4_energy = float(
            pyscf.dispersion.dftd4.DFTD4Dispersion(mol, xc=xc).get_dispersion()["energy"]
        )
        assert result.dispersion_energy_hartree is not None
        # Вклад D4 — главный предмет сверки: он обязан совпасть с
        # независимой сборкой в пределах REL_TOL.
        assert abs(result.dispersion_energy_hartree - d4_energy) <= REL_TOL * abs(d4_energy)
        # Полная сумма: электронный вклад — разность двух независимых SCF
        # (наш движок и PySCF, разные сетки интеграции для DFT), и её
        # расхождение (до ~1e-6 для PBE) не относится к D4; поэтому на
        # полную сумму даём отдельный, более крупный допуск и проверяем
        # электронный и дисперсионный вклады раздельно.
        our_scf = result.energy_hartree - result.dispersion_energy_hartree
        assert abs(our_scf - scf_energy) <= 2e-6
        assert abs(result.energy_hartree - (scf_energy + d4_energy)) <= 2e-6


def test_engine_optimization_with_d4_tracks_the_correction() -> None:
    """Оптимизация с D4: на каждом шаге градиент = SCF + D4, в итоге — полная энергия.

    Проверка независимая: вклад D4 в финальной геометрии пересчитывается
    ядром ``DFTD4`` отдельно от пайплайна и обязан совпасть с тем, что
    пайплайн положил в результат.
    """
    from quantumlab.domain.spec import (
        CalculationSpec,
        MethodSpec,
        OptimizationSpec,
        Task,
        TheoryFamily,
    )
    from quantumlab.engine.constants import angstrom_to_bohr
    from quantumlab.engine.contracts import EngineRequest
    from quantumlab.engine.dispersion_d4 import DFTD4
    from quantumlab.engine.reference import ReferenceEngine

    molecule = _water()
    spec = CalculationSpec(
        task=Task.OPTIMIZATION,
        method=MethodSpec(
            theory=TheoryFamily.HF,
            basis="sto-3g",
            dispersion=DispersionCorrection.D4,
        ),
        optimization=OptimizationSpec(max_steps=4),
    )
    result = ReferenceEngine().run(
        EngineRequest(job_id="d4-opt", spec=spec, molecule=molecule, threads=1)
    )
    assert result.optimization_steps is not None and result.optimization_steps > 0
    assert result.dispersion_energy_hartree is not None
    final = result.final_molecule
    assert final is not None
    assert final.atoms != molecule.atoms, "геометрия должна была измениться"

    zs = [a.z for a in final.atoms]
    pos = np.array([a.position for a in final.atoms]) * angstrom_to_bohr(1.0)
    kernel = DFTD4(zs, pos, "hf", total_charge=float(final.charge))
    assert abs(result.dispersion_energy_hartree - kernel.energy()) < 1e-12
    # Электронный вклад — независимый RHF в той же финальной геометрии:
    # разные реализации SCF совпадают в пределах 1e-6.
    mol = pyscf.gto.M(
        atom=[(a.symbol, list(a.position)) for a in final.atoms],
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    scf_energy = float(mol.RHF().run().e_tot)
    our_scf = result.energy_hartree - result.dispersion_energy_hartree
    assert abs(our_scf - scf_energy) <= 2e-6
    assert result.dispersion_energy_hartree < 0.0
