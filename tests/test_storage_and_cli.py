"""Локальное хранилище заданий и CLI — сквозная проверка вертикального среза.

CLI здесь проверяется как настоящий пользовательский путь: создать задание →
увидеть подобранные параметры → выполнить расчёт, если метод реализован, или
получить честный отказ, если нет → увидеть задание в списке → остановить его.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantumlab.cli import main
from quantumlab.domain.job import Job
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    PrecisionProfile,
    Task,
    TheoryFamily,
)
from quantumlab.engine.basis import build_basis
from quantumlab.engine.capabilities import Availability
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.registry import default_registry
from quantumlab.engine.scf import ScfSettings, run_rhf
from quantumlab.errors import InvalidJobTransitionError
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.storage.local_jobs import LocalJobStore

FIXTURES = Path(__file__).parent / "fixtures"
WATER = FIXTURES / "water.xyz"
HYDROGEN = FIXTURES / "hydrogen.xyz"

#: Жёсткие допуски: тесты сверяют число итераций, поэтому сходимость обязана
#: определяться физикой, а не слишком свободным порогом.
_TIGHT_SETTINGS = ScfSettings(energy_tolerance=1e-11, density_tolerance=1e-9, max_iterations=200)


def make_job(name: str = "water-opt") -> Job:
    return Job(
        name=name,
        project_id="proj-1",
        owner="tester",
        spec=CalculationSpec(task=Task.OPTIMIZATION, profile=PrecisionProfile.STANDARD),
        molecule_uri="artifact://molecules/water.xyz",
        molecule_hash="0" * 64,
    )


def test_job_round_trip(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "store")
    job = make_job()
    store.save(job)
    loaded = store.load(job.id)
    assert loaded.id == job.id
    assert loaded.spec.task is Task.OPTIMIZATION
    assert loaded.status is JobStatus.DRAFT
    assert len(store) == 1


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "store")
    store.save(make_job())
    leftovers = [path.name for path in store.jobs_dir.iterdir() if ".tmp-" in path.name]
    assert leftovers == []


def test_invalid_transition_is_not_persisted(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "store")
    job = make_job()
    store.save(job)

    def bad_update(item: Job) -> None:
        item.transition_to(JobStatus.COMPLETED)

    with pytest.raises(InvalidJobTransitionError):
        store.update(job.id, bad_update)
    assert store.load(job.id).status is JobStatus.DRAFT


def test_missing_job_raises_lookup_error(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "store")
    with pytest.raises(LookupError):
        store.load("no-such-id")


def test_cli_version_and_capabilities(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lang", "ru", "--data-dir", str(tmp_path), "version"]) == 0
    assert "QuantumLab 0.1.0" in capsys.readouterr().out

    assert (
        main(["--lang", "ru", "--data-dir", str(tmp_path), "capabilities", "--kind", "method"]) == 0
    )
    output = capsys.readouterr().out
    assert "method:ccsd_t" in output
    assert "not_implemented" in output


def test_cli_molecule_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--lang", "ru", "--data-dir", str(tmp_path), "molecule", "inspect", str(WATER)])
    assert code == 0
    output = capsys.readouterr().out
    assert "H2O" in output
    assert "Электронов: 10" in output
    assert "нарушений не найдено" in output


def test_cli_plan_explains_automatic_choices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "plan",
            str(WATER),
            "--task",
            "optimize",
            "--profile",
            "high-accuracy",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Высокая точность" in output
    assert "def2-tzvp" in output
    # Профиль обещает PBE0 + D3(BJ); PBE0 реализован, а дисперсионной поправки
    # нет, поэтому рекомендатель обязан сказать об этом вслух, а не выдать план,
    # который выполнится как расчёт без обещанной поправки (§54 ТЗ).
    assert "pbe0" in output
    assert "ещё не реализована" in output


def test_cli_runs_the_recommended_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Автоподобранный план обязан выполняться, а не падать после «Рассчитать».

    Раньше «Рекомендуемые настройки» выдавали DFT-функционал, которого в ядре
    нет, и расчёт умирал сразу после запуска. Теперь рекомендатель сверяется с
    реестром, поэтому recommended-план доходит до настоящего результата.
    Честность про недоступные методы проверяют соседние тесты.
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(HYDROGEN),
            "--task",
            "energy",
            "--profile",
            "screening",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Задание поставлено в очередь" in output
    assert "Энергия:" in output
    assert "Результат сохранён" in output
    # Никакого выдуманного числа: энергия получена настоящим SCF.
    assert "э" in output


def test_cli_run_executes_single_point_and_saves_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Реализованный путь выполняется по-настоящему и сохраняет результат.

    Это сквозная проверка вертикального среза: CLI → реестр → ядро → RHF →
    JSON-результат → статус задания. Энергия — та же, что подтверждена
    независимой сверкой с PySCF.
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(WATER),
            "--task",
            "energy",
            "--method",
            "hf",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Запускаю" in output
    assert "SCF сошёлся" in output
    assert "-74.9630296542" in output

    store = LocalJobStore(tmp_path)
    jobs = store.list()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status is JobStatus.COMPLETED
    assert job.result_uri is not None and job.result_uri.endswith(".json")

    payload = json.loads(store.result_path(job.id).read_text(encoding="utf-8"))
    assert payload["energy_hartree"] == pytest.approx(-74.9630296542, abs=1e-8)
    assert payload["converged"] is True
    assert payload["job_id"] == job.id
    assert payload["fingerprint"]["digest"]
    assert payload["environment"]["engine_backend"] == "numpy-dense-cpu"


def test_cli_run_warns_instead_of_hiding_basis_scheme(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Базис со сферической публикацией d даёт расчёт с предупреждением.

    Задание завершается как ``completed_with_warnings``: результат получен, но
    скрывать расхождение схемы с опубликованной было бы обманом.
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(WATER),
            "--task",
            "energy",
            "--method",
            "hf",
            "--basis",
            "def2-svp",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Предупреждения" in output
    assert "сферической" in output
    assert LocalJobStore(tmp_path).list()[0].status is JobStatus.COMPLETED_WITH_WARNINGS


def test_cli_run_reports_unavailable_task_honestly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """IRC требует пути по дну долины, которого нет: отказ, а не число.

    Задание остаётся в очереди — это правда: оно выполнится, когда появится
    соответствующее ядро.
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(WATER),
            "--task",
            "irc",
            "--method",
            "hf",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 1
    output = capsys.readouterr().out
    assert "Расчётное ядро пока не подключено" in output
    assert "Метод пока недоступен" in output
    store = LocalJobStore(tmp_path)
    assert store.list()[0].status is JobStatus.QUEUED
    # Результата быть не должно: расчёт не выполнялся.
    assert not store.result_path(store.list()[0].id).exists()


def test_cli_run_rejects_unimplemented_functional(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """B3LYP не реализован: запрос отклоняется до создания задания, а не в середине расчёта."""
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(WATER),
            "--task",
            "energy",
            "--method",
            "dft",
            "--functional",
            "tpssh",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 2
    output = capsys.readouterr()
    assert "Функционал не найден" in output.err
    # Никакого выдуманного числа в выводе быть не должно.
    assert "Энергия" not in output.out


def test_cli_run_optimization_saves_geometry_and_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Оптимизация выполняется по-настоящему и сохраняет обе сущности.

    Молекула — растянутый H₂: оптимизация обязана укоротить связь и сообщить
    число шагов. Результат и геометрия сохраняются раздельно: это разные
    артефакты, и у задания, меняющего структуру, должны быть оба.
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "run",
            str(HYDROGEN),
            "--task",
            "optimize",
            "--method",
            "hf",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Оптимизация геометрии сошлась" in output
    assert "Оптимизированная геометрия сохранена" in output

    store = LocalJobStore(tmp_path)
    job = store.list()[0]
    assert job.status is JobStatus.COMPLETED

    result = json.loads(store.result_path(job.id).read_text(encoding="utf-8"))
    assert result["optimization_steps"] >= 1
    assert result["final_molecule"] is not None

    saved = store.geometry_path(job.id).read_text(encoding="utf-8").splitlines()
    assert saved[0].strip() == "2"
    # Связь — расстояние между атомами, а не координата второго: в декартовой
    # оптимизации без закреплённого центра масс молекула свободно смещается,
    # поэтому отдельные координаты «плывут», хотя энергия от этого не зависит.
    first = [float(value) for value in saved[2].split()[1:4]]
    second = [float(value) for value in saved[3].split()[1:4]]
    bond = sum((a - b) ** 2 for a, b in zip(first, second, strict=True)) ** 0.5
    assert bond < 0.95  # связь укоротилась
    assert bond == pytest.approx(0.7122, abs=2e-3)


def test_cli_plan_shows_the_coordinate_system_in_expert_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Экспертный режим показывает систему координат — молчаливого выбора нет.

    Дефолт спецификации — избыточные внутренние координаты, которых в ядре
    нет; CLI подставляет декартовы и обязан это показать (§8 ТЗ).
    """
    code = main(
        [
            "--lang",
            "ru",
            "--data-dir",
            str(tmp_path),
            "plan",
            str(HYDROGEN),
            "--task",
            "optimize",
            "--method",
            "hf",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 0
    assert "Система координат оптимизации: cartesian" in capsys.readouterr().out


def test_cli_job_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Отмена, повтор и журнал состояний на задании, которое ещё не выполнено.

    Берётся задача без ядра (IRC): такое задание честно остаётся в очереди,
    и на нём можно проверять переходы состояний. Реализованный расчёт здесь не
    годится — он выполнился бы и закрыл задание до отмены.
    """
    base = ["--lang", "ru", "--data-dir", str(tmp_path)]
    main([*base, "run", str(WATER), "--task", "irc", "--method", "hf", "--basis", "sto-3g"])
    capsys.readouterr()

    assert main([*base, "job", "list"]) == 0
    listed = capsys.readouterr().out
    assert "В очереди" in listed

    store = LocalJobStore(tmp_path)
    job_id = store.list()[0].id

    assert main([*base, "job", "status", job_id]) == 0
    assert "В очереди" in capsys.readouterr().out

    assert main([*base, "job", "cancel", job_id]) == 0
    assert store.load(job_id).status is JobStatus.CANCELLED

    # Повтор теперь действительно исполняет задание, а не только меняет статус.
    # Метод (IRC) недоступен, поэтому возврат ненулевой. Задание остаётся в
    # очереди — это правда: оно выполнится, когда появится соответствующее ядро.
    assert main([*base, "job", "retry", job_id]) == 1
    assert "недоступен" in capsys.readouterr().out + capsys.readouterr().err
    reloaded = store.load(job_id)
    assert reloaded.status is JobStatus.QUEUED
    assert reloaded.attempt == 1

    assert main([*base, "job", "logs", job_id]) == 0
    assert "Журнал состояний" in capsys.readouterr().out


def test_cli_unknown_job_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--lang", "ru", "--data-dir", str(tmp_path), "job", "status", "missing"])
    assert code == 2
    assert "не найдено" in capsys.readouterr().err


def test_cli_works_in_english(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Те же сообщения по-английски; задача без ядра, чтобы не запускать расчёт."""
    code = main(
        [
            "--lang",
            "en",
            "--data-dir",
            str(tmp_path),
            "run",
            str(WATER),
            "--task",
            "irc",
            "--method",
            "hf",
            "--basis",
            "sto-3g",
        ]
    )
    assert code == 1
    output = capsys.readouterr().out
    assert "Job created" in output
    assert "The compute engine is not connected yet" in output
    assert "This method is not available yet" in output


def test_cli_rejects_unknown_task(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["--lang", "ru", "--data-dir", str(tmp_path), "plan", str(WATER), "--task", "teleport"]
    )
    assert code == 2
    assert "Неизвестная задача" in capsys.readouterr().err


def _queued_job(
    tmp_path: Path, *, final: JobStatus = JobStatus.FAILED
) -> tuple[LocalJobStore, str]:
    """Создаёт задание и доводит его до ``final`` штатными переходами."""
    store = LocalJobStore(tmp_path)
    molecule = Molecule.from_xyz(WATER.read_text(encoding="utf-8"), name="water")
    spec = CalculationSpec(
        task=Task.SINGLE_POINT,
        method=MethodSpec(theory=TheoryFamily.HF, basis="sto-3g"),
    )
    job = Job(
        name="water-test",
        project_id="p",
        owner="o",
        spec=spec,
        molecule_uri="file:///placeholder",
        molecule_hash=molecule.structure_hash(),
        resources=spec.resources,
    )
    store.store_molecule(job.id, molecule.to_xyz())
    store.save(job.model_copy(update={"molecule_uri": f"file://{store.molecule_path(job.id)}"}))
    path: tuple[JobStatus, ...] = (JobStatus.QUEUED, JobStatus.STARTING, JobStatus.RUNNING)
    if final is JobStatus.PAUSED:
        path = (*path, JobStatus.PAUSED)
    elif final is JobStatus.FAILED:
        path = (*path, JobStatus.FAILED)
    for status in path:
        _advance(store, job.id, status)
    return store, job.id


def _advance(store: LocalJobStore, job_id: str, status: JobStatus) -> None:
    """Один переход состояния.

    Вынесен из цикла намеренно: замыкание с аргументом по умолчанию mypy не
    выводит, а захватывать переменную цикла напрямую значило бы привязать все
    вызовы к последнему значению.
    """
    store.update(job_id, lambda item: item.transition_to(status, actor="test"))


def test_job_retry_actually_reruns_the_calculation(tmp_path: Path) -> None:
    """Повтор выполняет расчёт, а не только меняет статус.

    Прежняя реализация возвращала задание в очередь и печатала «повтор
    поставлен», но очередь никто не обрабатывал: задание висело в QUEUED без
    результата. Сообщение об успешном повторе при отсутствии расчёта — это
    ровно тот класс неправды, который запрещает §54 ТЗ.
    """
    store, job_id = _queued_job(tmp_path)
    assert store.load(job_id).status is JobStatus.FAILED

    code = main(["--data-dir", str(tmp_path), "job", "retry", job_id])

    after = store.load(job_id)
    assert code == 0
    assert after.attempt == 1
    assert after.status is JobStatus.COMPLETED
    assert after.result_uri is not None
    assert Path(store.result_path(job_id)).exists()


def test_job_resume_is_refused_while_checkpoints_do_not_exist(tmp_path: Path) -> None:
    """Без контрольной точки «продолжить» честно отказывает.

    Прежняя реализация переводила задание в RUNNING и печатала «Продолжен»,
    хотя не запускала ничего и продолжать было не с чего: пользователь видел бы
    расчёт, которого не существует. Статус обязан остаться прежним.
    """
    store, job_id = _queued_job(tmp_path, final=JobStatus.PAUSED)

    code = main(["--data-dir", str(tmp_path), "job", "resume", job_id])

    assert code == 2
    assert store.load(job_id).status is JobStatus.PAUSED, "статус не должен меняться"
    # Контрольные точки реализованы, но не для всех расчётов — поэтому
    # partial. Утверждение про «недоступно» стало бы неправдой.
    checkpoint = default_registry().get("job:checkpoint")
    assert checkpoint.availability is Availability.PARTIAL
    assert checkpoint.limitations, "partial без перечисленных ограничений — это «готово»"


def _paused_with_checkpoint(tmp_path: Path) -> tuple[LocalJobStore, str, int]:
    """Приостановленное задание с настоящей контрольной точкой.

    Контрольная точка создаётся настоящим движком, а не пишется в файл
    вручную: иначе тест проверял бы умение теста сочинять JSON, а не
    способность расчёта продолжиться. Статусы меняются штатными переходами.

    Возвращает и число итераций «с нуля» — без него нельзя отличить
    продолжение от повторного полного расчёта.
    """
    store, job_id = _queued_job(tmp_path, final=JobStatus.PAUSED)
    job = store.load(job_id)
    molecule = Molecule.from_xyz(
        store.molecule_path(job_id).read_text(encoding="utf-8"), name=job.name
    )
    baseline = run_rhf(build_basis("sto-3g", molecule), molecule, _TIGHT_SETTINGS)

    def persist(payload: str) -> None:
        reference = store.save_checkpoint(job_id, job.attempt, payload)
        store.update(job_id, lambda item: setattr(item, "checkpoint_uri", reference.uri))

    ReferenceEngine(default_registry()).run(
        EngineRequest(job_id=job_id, molecule=molecule, spec=job.spec),
        checkpoint_sink=persist,
    )
    assert store.load(job_id).status is JobStatus.PAUSED
    return store, job_id, baseline.iterations


def test_job_resume_continues_from_the_checkpoint(tmp_path: Path) -> None:
    """«Продолжить расчёт» продолжает, а не пересчитывает заново.

    Проверяется ровно то, что отличает продолжение от нового запуска: итераций
    заметно меньше, чем с нуля, а энергия та же. Команда, которая выполняла бы
    полный расчёт под видом продолжения, прошла бы проверку статуса, но не эту.
    """
    store, job_id, baseline = _paused_with_checkpoint(tmp_path)
    assert store.load(job_id).checkpoint_uri is not None

    code = main(["--data-dir", str(tmp_path), "job", "resume", job_id])

    after = store.load(job_id)
    assert code == 0
    assert after.status is JobStatus.COMPLETED
    result = json.loads(Path(store.result_path(job_id)).read_text(encoding="utf-8"))
    assert result["scf_iterations"] < baseline
    assert result["energy_hartree"] == pytest.approx(-74.9630296542, abs=1e-9)


def test_job_resume_detects_a_tampered_checkpoint(tmp_path: Path) -> None:
    """Подменённая контрольная точка обнаруживается по контрольной сумме.

    Без проверки расчёт сошёлся бы на изменённой плотности и выдал число,
    которому нельзя доверять, — при этом статус выглядел бы успешным. Задание
    обязано остаться приостановленным: оно не выполнено.
    """
    store, job_id, _ = _paused_with_checkpoint(tmp_path)
    path = store.checkpoint_path(job_id, 0)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["density"][0][0] += 0.25
    path.write_text(json.dumps(data), encoding="utf-8")

    code = main(["--data-dir", str(tmp_path), "job", "resume", job_id])

    assert code == 2
    assert store.load(job_id).status is JobStatus.PAUSED
    assert not Path(store.result_path(job_id)).exists()
