"""Контрольные точки SCF.

Контрольная точка — это не «статус задачи», а именно физическое состояние
расчёта: плотность, из которой SCF продолжает сходиться. Поэтому здесь
принципиальны две вещи, которые легко опустить и получить молча неверный ответ:

1. **Целостность.** Сохраняются отпечатки молекулы и базиса. Плотность,
   построенная в другой геометрии или в другом базисе, математически является
   матрицей того же размера, поэтому расчёт с ней сошёлся бы и выдал число —
   просто относящееся к другой задаче. Проверка отпечатков превращает такую
   подмену в явную ошибку.

2. **Валидность матрицы.** Плотность обязана быть симметричной и давать верное
   число электронов: ``tr(D·S) = N``. Повреждённый или обрезанный файл не должен
   превращаться в «расчёт, который сошёлся не туда».

Формат — обычный JSON: контрольные точки читаются человеком при разборе
падений, а объём для используемых сейчас базисов невелик. Переход на двоичный
формат потребуется вместе с большими базисами и записью на каждой итерации.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from quantumlab.domain.molecule import Molecule

#: Версия схемы. Любое изменение состава полей обязано её поднять: старый
#: читатель, встретив новое поле, не должен молча считать расчёт продолжимым.
CHECKPOINT_SCHEMA_VERSION = "1"

#: Схема для ссылок на артефакты.
CHECKPOINT_ARTIFACT_SCHEMA = f"quantumlab.checkpoint.v{CHECKPOINT_SCHEMA_VERSION}"

#: Допуск на число электронов. Не ноль: плотность хранится в JSON с конечной
#: точностью, а трассировка с матрицей перекрывания усиливает погрешность.
_ELECTRON_COUNT_TOLERANCE = 1e-6

#: Допуск на симметрию. Проверяется максимум модуля разности, а не норма,
#: чтобы единичный выброс в одном элементе не тонул в сумме.
_SYMMETRY_TOLERANCE = 1e-10


class CheckpointError(ValueError):
    """Контрольная точка непригодна к использованию.

    Отдельный тип нужен, чтобы вызывающая сторона различала «нет контрольной
    точки» (нормальная ситуация для нового задания) и «контрольная точка есть,
    но доверять ей нельзя» (требуется внимание человека).
    """


def molecule_fingerprint(molecule: Molecule) -> str:
    """Отпечаток молекулы: состав, заряд, кратность и геометрия.

    Берётся :meth:`~quantumlab.domain.molecule.Molecule.structure_hash`, а не
    хеш текста XYZ. Различие принципиально и найдено на сквозном прогоне:
    ``to_xyz`` встраивает в заголовок имя молекулы, а Job Manager перечитывает
    структуру под именем задания. Одинаковая химия с другой меткой давала бы
    другой отпечаток, и честный рестарт отклонялся бы как подмена.

    ``structure_hash`` округляет координаты до 1e-8 Å и не включает имя,
    поэтому отпечаток устойчив к формату файла и к переименованию, но меняется
    при перестановке атомов — а это правильно: порядок задаёт нумерацию
    базисных функций, а значит и смысл каждого элемента матрицы плотности.
    """
    return molecule.structure_hash()


@dataclass(frozen=True, slots=True)
class ScfCheckpoint:
    """Состояние SCF, достаточное для продолжения расчёта."""

    molecule_fingerprint: str
    basis: str
    density: np.ndarray
    total_energy: float
    iterations: int
    n_electrons: int

    def dump(self) -> str:
        """Сериализует в JSON.

        Плотность раскладывается в вложенные списки. Числа — через ``repr``
        floats в JSON, то есть с точностью до round-trip; этого достаточно,
        потому что рестарт — это начальное приближение, а не продолжение с
        побитово той же матрицы.
        """
        return json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "kind": "scf",
                "molecule_fingerprint": self.molecule_fingerprint,
                "basis": self.basis,
                "total_energy": self.total_energy,
                "iterations": self.iterations,
                "n_electrons": self.n_electrons,
                "density": self.density.tolist(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def write_scf_checkpoint(
    *,
    molecule: Molecule,
    basis: str,
    density: np.ndarray,
    total_energy: float,
    iterations: int,
) -> str:
    """Собирает контрольную точку из текущего состояния SCF."""
    return ScfCheckpoint(
        molecule_fingerprint=molecule_fingerprint(molecule),
        basis=basis,
        density=np.asarray(density, dtype=float),
        total_energy=float(total_energy),
        iterations=int(iterations),
        n_electrons=molecule.n_electrons,
    ).dump()


def read_scf_checkpoint(payload: str, *, overlap: np.ndarray) -> ScfCheckpoint:
    """Читает контрольную точку и проверяет, что ею можно пользоваться.

    Параметр ``overlap`` — матрица перекрывания целевого расчёта: без неё нельзя
    проверить ни размерность, ни число электронов, то есть проверить нечего.

    Выбрасывает :class:`CheckpointError` при любом расхождении. Возвращать
    непроверенную плотность нельзя: рестарт с ней сошёлся бы и выдал число,
    принадлежащее другой задаче.
    """
    try:
        data: object = json.loads(payload)
    except json.JSONDecodeError as error:
        msg = f"Контрольная точка не является корректным JSON: {error}"
        raise CheckpointError(msg) from error

    if not isinstance(data, dict):
        msg = "Контрольная точка должна быть объектом JSON"
        raise CheckpointError(msg)

    version = data.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        msg = (
            f"Контрольная точка схемы {version!r}, ожидается "
            f"{CHECKPOINT_SCHEMA_VERSION!r}. Продолжать расчёт по старой схеме "
            "нельзя: состав полей мог измениться."
        )
        raise CheckpointError(msg)

    if data.get("kind") != "scf":
        msg = f"Ожидалась контрольная точка SCF, получена {data.get('kind')!r}"
        raise CheckpointError(msg)

    density_raw = data.get("density")
    if not isinstance(density_raw, list):
        msg = "В контрольной точке отсутствует матрица плотности"
        raise CheckpointError(msg)

    try:
        density = np.asarray(density_raw, dtype=float)
    except (TypeError, ValueError) as error:
        msg = f"Матрица плотности содержит нечисловые значения: {error}"
        raise CheckpointError(msg) from error

    if density.shape != overlap.shape:
        msg = (
            f"Размер матрицы плотности {density.shape} не соответствует базису "
            f"целевого расчёта {overlap.shape}. Скорее всего, контрольная точка "
            "от другого базиса."
        )
        raise CheckpointError(msg)

    asymmetry = float(np.max(np.abs(density - density.T)))
    if asymmetry > _SYMMETRY_TOLERANCE:
        msg = (
            f"Матрица плотности несимметрична (максимальное расхождение {asymmetry:.3e}). "
            "Файл повреждён, рестарт с такой матрицы недопустим."
        )
        raise CheckpointError(msg)

    stored_electrons = data.get("n_electrons")
    if not isinstance(stored_electrons, int):
        msg = "В контрольной точке отсутствует или повреждено число электронов"
        raise CheckpointError(msg)

    electrons = float(np.trace(density @ overlap))
    if abs(electrons - stored_electrons) > _ELECTRON_COUNT_TOLERANCE:
        msg = (
            f"Контрольная точка описывает {stored_electrons} электронов, но "
            f"tr(D·S) = {electrons:.6f}. Матрица не согласована с сохранённым "
            "состоянием."
        )
        raise CheckpointError(msg)

    energy = data.get("total_energy")
    iterations = data.get("iterations")
    if not isinstance(energy, int | float) or not isinstance(iterations, int):
        msg = "В контрольной точке повреждены энергия или число итераций"
        raise CheckpointError(msg)

    return ScfCheckpoint(
        molecule_fingerprint=str(data.get("molecule_fingerprint")),
        basis=str(data.get("basis")),
        density=density,
        total_energy=float(energy),
        iterations=int(iterations),
        n_electrons=stored_electrons,
    )


def assert_matches_job(checkpoint: ScfCheckpoint, *, molecule: Molecule, basis: str) -> None:
    """Проверяет, что контрольная точка относится именно к этому расчёту."""
    expected = molecule_fingerprint(molecule)
    if checkpoint.molecule_fingerprint != expected:
        msg = (
            "Контрольная точка принадлежит другой геометрии или другой молекуле. "
            "Рестарт с ней сошёлся бы, но описывал бы другую систему."
        )
        raise CheckpointError(msg)
    if checkpoint.basis != basis:
        msg = (
            f"Контрольная точка построена в базисе {checkpoint.basis!r}, а расчёт "
            f"запрошен в {basis!r}. Плотность в другом базисе неприменима."
        )
        raise CheckpointError(msg)


def payload_sha256(payload: str) -> str:
    """Контрольная сумма содержимого — для ссылки на артефакт."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_URI_SHA256_MARKER = "#sha256="


def checkpoint_uri(filename: str, digest: str) -> str:
    """Собирает URI артефакта с контрольной суммой в фрагменте.

    Сумма входит в URI, а не остаётся отдельным полем: так её невозможно
    потерять при переносе ссылки между слоями, и проверка целостности доступна
    любому, у кого есть только строка ``checkpoint_uri``.
    """
    return f"artifact://checkpoints/{filename}{_URI_SHA256_MARKER}{digest}"


def sha256_from_uri(uri: str) -> str | None:
    """Достаёт контрольную сумму из URI, собранного :func:`checkpoint_uri`."""
    marker = _URI_SHA256_MARKER
    index = uri.rfind(marker)
    if index < 0:
        return None
    return uri[index + len(marker) :]
