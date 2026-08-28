"""Контракт контрольных точек SCF.

Контрольная точка — это обещание продолжить расчёт. Проверяются три вещи, без
которых такое обещание было бы ложью:

* рестарт действительно даёт то же число, а не «похожее»;
* непригодная контрольная точка отклоняется, а не подставляется молча;
* подмена содержимого обнаруживается.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantumlab.domain.molecule import Molecule
from quantumlab.engine.basis import build_basis
from quantumlab.engine.checkpoint import (
    CheckpointError,
    assert_matches_job,
    molecule_fingerprint,
    payload_sha256,
    read_scf_checkpoint,
    sha256_from_uri,
    write_scf_checkpoint,
)
from quantumlab.engine.scf import ScfSettings, build_integrals, run_rhf

FIXTURES = Path(__file__).parent / "fixtures"
WATER = FIXTURES / "water.xyz"
HYDROGEN = FIXTURES / "hydrogen.xyz"
TIGHT = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


@pytest.fixture(scope="module")
def water() -> Molecule:
    """Молекула воды из фикстуры."""
    return Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")


def test_restart_reaches_the_same_energy_in_fewer_iterations(water: Molecule) -> None:
    """Рестарт из контрольной точки даёт то же число за меньшее число итераций.

    Оба условия обязательны по отдельным причинам. Совпадение энергии без
    выигрыша означало бы, что плотность проигнорирована и расчёт начат заново.
    Выигрыш без совпадения означал бы, что продолжался другой расчёт.
    """
    basis = build_basis("sto-3g", water)
    prepared = build_integrals(basis, water)

    fresh = run_rhf(basis, water, TIGHT, integrals=prepared)
    payload = write_scf_checkpoint(
        molecule=water,
        basis="sto-3g",
        density=fresh.density,
        total_energy=fresh.total_energy,
        iterations=fresh.iterations,
    )
    state = read_scf_checkpoint(payload, overlap=prepared.overlap)
    assert_matches_job(state, molecule=water, basis="sto-3g")

    resumed = run_rhf(basis, water, TIGHT, integrals=prepared, initial_density=state.density)

    assert resumed.iterations < fresh.iterations
    assert resumed.total_energy == pytest.approx(fresh.total_energy, abs=1e-11)
    assert resumed.strategies_used == ("checkpoint-restart",)


def test_checkpoint_does_not_depend_on_the_molecule_label(water: Molecule) -> None:
    """Отпечаток не зависит от имени молекулы.

    Найдено на сквозном прогоне: ``to_xyz`` встраивает имя в заголовок, а Job
    Manager перечитывает структуру под именем задания. Если бы отпечаток
    считался по тексту XYZ, честный рестарт отклонялся бы как подмена.
    """
    basis = build_basis("sto-3g", water)
    prepared = build_integrals(basis, water)
    fresh = run_rhf(basis, water, TIGHT, integrals=prepared)
    payload = write_scf_checkpoint(
        molecule=water,
        basis="sto-3g",
        density=fresh.density,
        total_energy=fresh.total_energy,
        iterations=fresh.iterations,
    )
    state = read_scf_checkpoint(payload, overlap=prepared.overlap)

    renamed = water.model_copy(update={"name": "другое-имя"})
    assert molecule_fingerprint(renamed) == molecule_fingerprint(water)
    assert_matches_job(state, molecule=renamed, basis="sto-3g")


def test_checkpoint_from_another_system_is_rejected(water: Molecule) -> None:
    """Чужая геометрия или чужой базис отклоняются, а не подставляются.

    Матрица плотности от другой молекулы часто имеет тот же размер, поэтому без
    проверки расчёт сошёлся бы и выдал число, описывающее другую систему.
    """
    basis = build_basis("sto-3g", water)
    prepared = build_integrals(basis, water)
    fresh = run_rhf(basis, water, TIGHT, integrals=prepared)
    payload = write_scf_checkpoint(
        molecule=water,
        basis="sto-3g",
        density=fresh.density,
        total_energy=fresh.total_energy,
        iterations=fresh.iterations,
    )
    state = read_scf_checkpoint(payload, overlap=prepared.overlap)

    hydrogen = Molecule.from_xyz(HYDROGEN.read_text(encoding="utf-8"), name="h2")
    with pytest.raises(CheckpointError, match="другой геометрии"):
        assert_matches_job(state, molecule=hydrogen, basis="sto-3g")
    with pytest.raises(CheckpointError, match="базисе"):
        assert_matches_job(state, molecule=water, basis="6-31g")


def test_corrupted_density_is_rejected(water: Molecule) -> None:
    """Повреждённая матрица плотности не превращается в «считаем дальше».

    Проверяются оба инварианта, которые читатель обязан контролировать:
    симметрия и число электронов. Они нарушаются типичными повреждениями —
    подменой значений и обрезкой файла.
    """
    basis = build_basis("sto-3g", water)
    prepared = build_integrals(basis, water)
    fresh = run_rhf(basis, water, TIGHT, integrals=prepared)
    payload = write_scf_checkpoint(
        molecule=water,
        basis="sto-3g",
        density=fresh.density,
        total_energy=fresh.total_energy,
        iterations=fresh.iterations,
    )

    asymmetric = json.loads(payload)
    asymmetric["density"][0][1] += 0.5
    with pytest.raises(CheckpointError, match="несимметрична"):
        read_scf_checkpoint(json.dumps(asymmetric), overlap=prepared.overlap)

    # Матрица остаётся корректной, но перестаёт соответствовать сохранённому
    # описанию: заявлено другое число электронов.
    wrong_count = json.loads(payload)
    wrong_count["n_electrons"] = 9
    with pytest.raises(CheckpointError, match="электронов"):
        read_scf_checkpoint(json.dumps(wrong_count), overlap=prepared.overlap)

    # И схема: старая версия не должна читаться новой, потому что состав полей
    # мог измениться.
    old_schema = json.loads(payload)
    old_schema["schema_version"] = "0"
    with pytest.raises(CheckpointError, match="схемы"):
        read_scf_checkpoint(json.dumps(old_schema), overlap=prepared.overlap)


def test_checksum_round_trip() -> None:
    """Контрольная сумма переживает путь через URI артефакта.

    Сумма входит в URI именно для того, чтобы её нельзя было потерять при
    передаче ссылки между слоями. Если извлечение перестанет работать, проверка
    целостности молча выключится.
    """
    payload = '{"schema_version": "1"}'
    digest = payload_sha256(payload)
    uri = f"artifact://checkpoints/job-0.json#sha256={digest}"
    assert sha256_from_uri(uri) == digest
    assert sha256_from_uri("artifact://checkpoints/job-0.json") is None


def test_initial_density_with_wrong_shape_is_rejected(water: Molecule) -> None:
    """Плотность не того размера отклоняется движком на входе."""
    basis = build_basis("sto-3g", water)
    with pytest.raises(ValueError, match="Начальная плотность"):
        run_rhf(basis, water, TIGHT, initial_density=np.zeros((3, 3)))
