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
from quantumlab.engine.optimizer import OptimizationStep, OptimizerState

#: Версия схемы. Любое изменение состава полей обязано её поднять: старый
#: читатель, встретив новое поле, не должен молча считать расчёт продолжимым.
CHECKPOINT_SCHEMA_VERSION = "2"

#: Схемы, которые читатель принимает. Версия 1 — только полная плотность
#: (замкнутая оболочка); версия 2 добавляет необязательные спиновые плотности
#: ``density_alpha``/``density_beta`` для UHF и UKS. Поля версии 1 не менялись,
#: поэтому старые контрольные точки остаются пригодными для рестарта RHF/RKS.
_READABLE_SCHEMA_VERSIONS = ("1", "2")

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
    density_alpha: np.ndarray | None = None
    density_beta: np.ndarray | None = None
    n_alpha: int | None = None
    n_beta: int | None = None

    @property
    def is_spin_resolved(self) -> bool:
        """Есть ли отдельные плотности α и β (контрольная точка UHF/UKS)."""
        return self.density_alpha is not None and self.density_beta is not None

    def dump(self) -> str:
        """Сериализует в JSON.

        Плотность раскладывается в вложенные списки. Числа — через ``repr``
        floats в JSON, то есть с точностью до round-trip; этого достаточно,
        потому что рестарт — это начальное приближение, а не продолжение с
        побитово той же матрицы.
        """
        payload: dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "scf",
            "molecule_fingerprint": self.molecule_fingerprint,
            "basis": self.basis,
            "total_energy": self.total_energy,
            "iterations": self.iterations,
            "n_electrons": self.n_electrons,
            "density": self.density.tolist(),
        }
        if self.density_alpha is not None and self.density_beta is not None:
            payload["density_alpha"] = self.density_alpha.tolist()
            payload["density_beta"] = self.density_beta.tolist()
            payload["n_alpha"] = self.n_alpha
            payload["n_beta"] = self.n_beta
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def write_scf_checkpoint(
    *,
    molecule: Molecule,
    basis: str,
    density: np.ndarray,
    total_energy: float,
    iterations: int,
    spin_densities: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    """Собирает контрольную точку из текущего состояния SCF.

    ``spin_densities`` — пара ``(D^α, D^β)`` для открытой оболочки; в этом случае
    ``density`` обязана быть их суммой (проверяется при чтении).
    """
    if spin_densities is None:
        return ScfCheckpoint(
            molecule_fingerprint=molecule_fingerprint(molecule),
            basis=basis,
            density=np.asarray(density, dtype=float),
            total_energy=float(total_energy),
            iterations=int(iterations),
            n_electrons=molecule.n_electrons,
        ).dump()
    n_unpaired = molecule.multiplicity - 1
    return ScfCheckpoint(
        molecule_fingerprint=molecule_fingerprint(molecule),
        basis=basis,
        density=np.asarray(density, dtype=float),
        total_energy=float(total_energy),
        iterations=int(iterations),
        n_electrons=molecule.n_electrons,
        density_alpha=np.asarray(spin_densities[0], dtype=float),
        density_beta=np.asarray(spin_densities[1], dtype=float),
        n_alpha=(molecule.n_electrons + n_unpaired) // 2,
        n_beta=(molecule.n_electrons - n_unpaired) // 2,
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
    if version not in _READABLE_SCHEMA_VERSIONS:
        msg = (
            f"Контрольная точка схемы {version!r}, ожидается одна из "
            f"{_READABLE_SCHEMA_VERSIONS!r}. Продолжать расчёт по старой схеме "
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

    spin = _read_spin_densities(data, overlap=overlap, total=density)
    return ScfCheckpoint(
        molecule_fingerprint=str(data.get("molecule_fingerprint")),
        basis=str(data.get("basis")),
        density=density,
        total_energy=float(energy),
        iterations=int(iterations),
        n_electrons=stored_electrons,
        density_alpha=spin[0] if spin is not None else None,
        density_beta=spin[1] if spin is not None else None,
        n_alpha=spin[2] if spin is not None else None,
        n_beta=spin[3] if spin is not None else None,
    )


def _read_spin_densities(
    data: dict[str, object], *, overlap: np.ndarray, total: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Читает и проверяет спиновые плотности; ``None`` — их в файле нет.

    Присутствие только одной из двух матриц — повреждение, а не «замкнутая
    оболочка»: тихо вернуть ``None`` значило бы продолжить расчёт открытой
    оболочки с чужой плотностью.
    """
    raw_alpha = data.get("density_alpha")
    raw_beta = data.get("density_beta")
    if raw_alpha is None and raw_beta is None:
        return None
    if raw_alpha is None or raw_beta is None:
        msg = "В контрольной точке есть только одна из спиновых плотностей α/β"
        raise CheckpointError(msg)
    n_alpha = data.get("n_alpha")
    n_beta = data.get("n_beta")
    if not isinstance(n_alpha, int) or not isinstance(n_beta, int):
        msg = "В контрольной точке отсутствуют или повреждены числа электронов α и β"
        raise CheckpointError(msg)
    matrices: list[np.ndarray] = []
    for label, raw, count in (("α", raw_alpha, n_alpha), ("β", raw_beta, n_beta)):
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            msg = f"Плотность {label} содержит нечисловые значения: {error}"
            raise CheckpointError(msg) from error
        if matrix.shape != overlap.shape:
            msg = f"Размер плотности {label} {matrix.shape} не соответствует базису {overlap.shape}"
            raise CheckpointError(msg)
        if float(np.max(np.abs(matrix - matrix.T))) > _SYMMETRY_TOLERANCE:
            msg = f"Плотность {label} несимметрична: файл повреждён"
            raise CheckpointError(msg)
        electrons = float(np.trace(matrix @ overlap))
        if abs(electrons - count) > _ELECTRON_COUNT_TOLERANCE:
            msg = f"Плотность {label}: tr(D·S) = {electrons:.6f}, ожидалось {count}"
            raise CheckpointError(msg)
        matrices.append(matrix)
    if float(np.max(np.abs(matrices[0] + matrices[1] - total))) > 1e-8:
        msg = "Полная плотность не равна сумме плотностей α и β: файл повреждён"
        raise CheckpointError(msg)
    return matrices[0], matrices[1], n_alpha, n_beta


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


# --------------------------------------------------------------------------- #
# Контрольная точка оптимизации геометрии
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OptimizationCheckpoint:
    """Состояние оптимизации геометрии, достаточное для продолжения.

    В отличие от SCF-точки здесь важна не плотность, а то, чего не восстановить
    пересчётом: **приближение гессиана** (оно накоплено из всех прошлых шагов —
    потеряв его, оптимизация начала бы с единичной матрицы и повторила бы
    половину пути), шаг и градиент для следующего BFGS-обновления и журнал.

    Плотность SCF сознательно не хранится: следующая энергия считается в новой
    геометрии, а плотность из старой не проходит проверку ``tr(D·S) = N``.
    """

    initial_fingerprint: str
    basis: str
    spec_hash: str
    state: OptimizerState


def spec_hash(canonical_json: str) -> str:
    """Отпечаток спецификации расчёта (метод, базис, сетка, оптимизация).

    Градиент и гессиан относятся к конкретной поверхности потенциальной энергии:
    продолжить с ними под другим функционалом или базисом значило бы получить
    траекторию по смеси двух поверхностей.
    """
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def write_optimization_checkpoint(
    *, molecule: Molecule, basis: str, spec_digest: str, state: OptimizerState
) -> str:
    """Сериализует состояние оптимизатора. ``molecule`` — исходная структура задания."""

    def vector(values: np.ndarray | None) -> list[float] | None:
        return None if values is None else np.asarray(values, dtype=float).tolist()

    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "optimization",
        "initial_fingerprint": molecule_fingerprint(molecule),
        "basis": basis,
        "spec_hash": spec_digest,
        "step_index": state.step_index,
        "coordinates": vector(state.coordinates),
        "energy_hartree": float(state.energy_hartree),
        "gradient": vector(state.gradient),
        "hessian": np.asarray(state.hessian, dtype=float).tolist(),
        "previous_step": vector(state.previous_step),
        "previous_gradient": vector(state.previous_gradient),
        "displacement": vector(state.displacement),
        "history": [
            {
                "index": step.index,
                "energy_hartree": step.energy_hartree,
                "max_force": step.max_force,
                "rms_force": step.rms_force,
                "max_displacement": step.max_displacement,
                "rms_displacement": step.rms_displacement,
            }
            for step in state.history
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def read_optimization_checkpoint(payload: str, *, molecule: Molecule) -> OptimizationCheckpoint:
    """Читает контрольную точку оптимизации и проверяет согласованность размеров.

    ``molecule`` — исходная структура задания: по ней проверяются число атомов
    и размер гессиана. Выбрасывает :class:`CheckpointError` при любом
    расхождении или повреждении.
    """
    try:
        data: object = json.loads(payload)
    except json.JSONDecodeError as error:
        msg = f"Контрольная точка не является корректным JSON: {error}"
        raise CheckpointError(msg) from error
    if not isinstance(data, dict):
        msg = "Контрольная точка должна быть объектом JSON"
        raise CheckpointError(msg)
    if data.get("schema_version") not in _READABLE_SCHEMA_VERSIONS:
        msg = f"Неизвестная схема контрольной точки: {data.get('schema_version')!r}"
        raise CheckpointError(msg)
    if data.get("kind") != "optimization":
        msg = f"Ожидалась контрольная точка оптимизации, получена {data.get('kind')!r}"
        raise CheckpointError(msg)

    n_cartesian = 3 * molecule.n_atoms

    def array(name: str, *, optional: bool = False) -> np.ndarray | None:
        raw = data.get(name)
        if raw is None and optional:
            return None
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as error:
            msg = f"Поле {name!r} контрольной точки повреждено: {error}"
            raise CheckpointError(msg) from error
        if not np.all(np.isfinite(values)):
            msg = f"Поле {name!r} содержит нечисловые значения"
            raise CheckpointError(msg)
        return values

    coordinates = array("coordinates")
    gradient = array("gradient")
    hessian = array("hessian")
    displacement = array("displacement")
    previous_step = array("previous_step", optional=True)
    previous_gradient = array("previous_gradient", optional=True)
    assert coordinates is not None
    assert gradient is not None
    assert hessian is not None
    assert displacement is not None
    for name, values in (
        ("coordinates", coordinates),
        ("gradient", gradient),
        ("displacement", displacement),
    ):
        if values.shape != (n_cartesian,):
            msg = f"Размер поля {name!r} {values.shape} не соответствует {molecule.n_atoms} атомам"
            raise CheckpointError(msg)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1] or hessian.shape[0] > n_cartesian:
        msg = f"Некорректная форма гессиана {hessian.shape}"
        raise CheckpointError(msg)
    if float(np.max(np.abs(hessian - hessian.T))) > 1e-8:
        msg = "Гессиан в контрольной точке несимметричен: файл повреждён"
        raise CheckpointError(msg)
    # Предыдущий градиент — декартов (n_cartesian) либо во внутренних координатах
    # (размер гессиана); какой именно, проверяет оптимизатор по своей системе координат.
    if previous_gradient is not None and previous_gradient.shape not in (
        (n_cartesian,),
        (hessian.shape[0],),
    ):
        msg = "Размер предыдущего градиента не соответствует задаче"
        raise CheckpointError(msg)
    if previous_step is not None and previous_step.shape != (hessian.shape[0],):
        msg = "Размер предыдущего шага не соответствует гессиану"
        raise CheckpointError(msg)

    step_index = data.get("step_index")
    energy = data.get("energy_hartree")
    raw_history = data.get("history")
    if (
        not isinstance(step_index, int)
        or not isinstance(energy, int | float)
        or not isinstance(raw_history, list)
    ):
        msg = "В контрольной точке повреждены номер шага, энергия или журнал"
        raise CheckpointError(msg)
    try:
        history = tuple(
            OptimizationStep(
                index=int(item["index"]),
                energy_hartree=float(item["energy_hartree"]),
                max_force=float(item["max_force"]),
                rms_force=float(item["rms_force"]),
                max_displacement=_optional_float(item["max_displacement"]),
                rms_displacement=_optional_float(item["rms_displacement"]),
            )
            for item in raw_history
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = f"Журнал итераций в контрольной точке повреждён: {error}"
        raise CheckpointError(msg) from error
    if len(history) != step_index + 1:
        msg = (
            f"Журнал содержит {len(history)} записей, а шаг — {step_index}: "
            "контрольная точка не согласована"
        )
        raise CheckpointError(msg)

    return OptimizationCheckpoint(
        initial_fingerprint=str(data.get("initial_fingerprint")),
        basis=str(data.get("basis")),
        spec_hash=str(data.get("spec_hash")),
        state=OptimizerState(
            step_index=step_index,
            coordinates=coordinates,
            energy_hartree=float(energy),
            gradient=gradient,
            hessian=hessian,
            previous_step=previous_step,
            previous_gradient=previous_gradient,
            displacement=displacement,
            history=history,
        ),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def assert_optimization_matches_job(
    checkpoint: OptimizationCheckpoint, *, molecule: Molecule, basis: str, spec_digest: str
) -> None:
    """Проверяет, что точка принадлежит именно этой оптимизации.

    Сверяется **исходная** структура задания, а не текущая: текущая — это
    результат работы, и по ней продолжение не отличило бы своё задание от чужого.
    """
    if checkpoint.initial_fingerprint != molecule_fingerprint(molecule):
        msg = "Контрольная точка оптимизации относится к другой исходной структуре"
        raise CheckpointError(msg)
    if checkpoint.basis != basis:
        msg = f"Контрольная точка построена в базисе {checkpoint.basis!r}, запрошен {basis!r}"
        raise CheckpointError(msg)
    if checkpoint.spec_hash != spec_digest:
        msg = (
            "Спецификация расчёта изменилась с момента записи контрольной точки: "
            "градиент и гессиан относятся к другой поверхности"
        )
        raise CheckpointError(msg)
