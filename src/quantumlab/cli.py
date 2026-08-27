"""Командная строка QuantumLab (§20 ТЗ).

CLI использует **тот же backend**, что GUI и REST API: доменные модели,
автоподбор параметров и реестр возможностей. Здесь нет отдельной «CLI-логики» —
иначе поведение интерфейсов неизбежно разъехалось бы.

Примеры::

    quantumlab molecule inspect benzene.xyz
    quantumlab plan benzene.xyz --task optimization --profile high-accuracy
    quantumlab run benzene.xyz --task optimize --profile standard
    quantumlab job list
    quantumlab job status <id>
    quantumlab job cancel <id>
    quantumlab capabilities --kind method
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from quantumlab.domain.job import Job
from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import (
    CalculationSpec,
    MethodSpec,
    PrecisionProfile,
    Task,
    TheoryFamily,
)
from quantumlab.engine.capabilities import CapabilityKind
from quantumlab.engine.contracts import EngineRequest
from quantumlab.engine.reference import ReferenceEngine
from quantumlab.engine.registry import CapabilityRegistry, default_registry
from quantumlab.errors import QuantumLabError
from quantumlab.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, t
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.recommend.profiles import HardwareContext, Resolution, resolve_profile
from quantumlab.storage.local_jobs import LocalJobStore
from quantumlab.version import __version__, api_version

_PROFILE_BY_CLI_NAME: dict[str, PrecisionProfile] = {
    "screening": PrecisionProfile.SCREENING,
    "fast": PrecisionProfile.SCREENING,
    "standard": PrecisionProfile.STANDARD,
    "high-accuracy": PrecisionProfile.HIGH_ACCURACY,
    "high": PrecisionProfile.HIGH_ACCURACY,
    "research": PrecisionProfile.RESEARCH,
}

_TASK_BY_CLI_NAME: dict[str, Task] = {
    "energy": Task.SINGLE_POINT,
    "single-point": Task.SINGLE_POINT,
    "optimize": Task.OPTIMIZATION,
    "optimization": Task.OPTIMIZATION,
    "freq": Task.FREQUENCIES,
    "frequencies": Task.FREQUENCIES,
    "ts": Task.TS_OPTIMIZATION,
    "irc": Task.IRC,
    "scan": Task.SCAN_1D,
    "properties": Task.PROPERTIES,
}


def build_parser() -> argparse.ArgumentParser:
    """Собирает парсер аргументов CLI."""
    parser = argparse.ArgumentParser(
        prog="quantumlab",
        description="QuantumLab — квантовохимические расчёты без языка input-файлов",
    )
    parser.add_argument(
        "--lang",
        choices=SUPPORTED_LOCALES,
        default=DEFAULT_LOCALE,
        help="язык вывода (по умолчанию русский)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".quantumlab",
        help="каталог локального хранилища заданий",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="версия пакета и контракта API")

    capabilities_parser = subparsers.add_parser("capabilities", help="что реально реализовано")
    capabilities_parser.add_argument(
        "--kind",
        choices=[
            "method",
            "functional",
            "basis",
            "task",
            "format",
            "backend",
            "scheduler",
            "property",
        ],
        default=None,
        help="категория возможностей",
    )

    molecule_parser = subparsers.add_parser("molecule", help="работа со структурой")
    molecule_sub = molecule_parser.add_subparsers(dest="molecule_command", required=True)
    inspect_parser = molecule_sub.add_parser("inspect", help="разобрать файл и проверить структуру")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--charge", type=int, default=0)
    inspect_parser.add_argument("--multiplicity", type=int, default=1)

    plan_parser = subparsers.add_parser("plan", help="показать подобранные параметры без запуска")
    _add_calculation_arguments(plan_parser)

    run_parser = subparsers.add_parser("run", help="создать и поставить в очередь расчёт")
    _add_calculation_arguments(run_parser)
    run_parser.add_argument("--name", default=None, help="имя задания")
    run_parser.add_argument("--project", default="default", help="идентификатор проекта")
    run_parser.add_argument("--owner", default="cli", help="пользователь")

    job_parser = subparsers.add_parser("job", help="управление заданиями")
    job_sub = job_parser.add_subparsers(dest="job_command", required=True)
    list_parser = job_sub.add_parser("list", help="список заданий")
    list_parser.add_argument("--status", default=None, help="фильтр по статусу")
    for name in ("status", "logs", "cancel", "resume", "retry"):
        item = job_sub.add_parser(name, help=f"команда «{name}» для задания")
        item.add_argument("job_id")

    return parser


def _add_calculation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="файл структуры в формате XYZ")
    parser.add_argument("--task", default="optimize", help="тип задачи")
    parser.add_argument("--profile", default="standard", help="профиль точности")
    parser.add_argument("--method", default=None, help="явный метод: hf | dft | mp2 | …")
    parser.add_argument("--functional", default=None, help="явный функционал, например pbe0")
    parser.add_argument("--basis", default=None, help="явный базис, например def2-tzvp")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--cores", type=int, default=8, help="доступное число ядер")
    parser.add_argument("--memory-mb", type=int, default=16384, help="доступная память, МБ")
    parser.add_argument("--gpus", type=int, default=0, help="число доступных GPU")


def _load_molecule(path: Path, *, charge: int, multiplicity: int) -> Molecule:
    return Molecule.from_xyz(path.read_text(encoding="utf-8"), name=path.stem).model_copy(
        update={"charge": charge, "multiplicity": multiplicity}
    )


def _build_spec(args: argparse.Namespace, registry: CapabilityRegistry) -> CalculationSpec:
    task = _TASK_BY_CLI_NAME.get(args.task.lower())
    if task is None:
        available = ", ".join(sorted(_TASK_BY_CLI_NAME))
        msg = f"Неизвестная задача {args.task!r}. Доступны: {available}"
        raise ValueError(msg)

    if args.method or args.functional or args.basis:
        method = MethodSpec(
            theory=TheoryFamily((args.method or "dft").lower()),
            functional=args.functional,
            basis=args.basis or "def2-svp",
        )
        registry.assert_available(f"basis:{method.basis}")
        if method.functional:
            registry.assert_available(f"functional:{method.functional}")
        return CalculationSpec(task=task, profile=None, method=method)

    profile = _PROFILE_BY_CLI_NAME.get(args.profile.lower())
    if profile is None:
        available = ", ".join(sorted(_PROFILE_BY_CLI_NAME))
        msg = f"Неизвестный профиль {args.profile!r}. Доступны: {available}"
        raise ValueError(msg)
    return CalculationSpec(task=task, profile=profile)


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата процесса."""
    args = build_parser().parse_args(argv)
    locale: str = args.lang
    registry = default_registry()

    try:
        if args.command == "version":
            print(t("cli.version", locale, version=__version__, api=api_version))
            return 0
        if args.command == "capabilities":
            return _command_capabilities(args, registry, locale)
        if args.command == "molecule":
            return _command_molecule(args, locale)
        if args.command == "plan":
            return _command_plan(args, registry, locale)
        if args.command == "run":
            return _command_run(args, registry, locale)
        if args.command == "job":
            return _command_job(args, locale)
    except QuantumLabError as error:
        print(f"{t('cli.error.header', locale)}: {error.explain(locale)}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as error:
        print(f"{t('cli.error.header', locale)}: {error}", file=sys.stderr)
        return 2

    msg = f"Неизвестная команда {args.command!r}"
    raise AssertionError(msg)


def _command_capabilities(
    args: argparse.Namespace, registry: CapabilityRegistry, locale: str
) -> int:
    print(t("cli.capabilities.title", locale))
    print(t("cli.capabilities.legend", locale))
    print()
    kind = CapabilityKind(args.kind) if args.kind else None
    capabilities = registry.list_capabilities(kind)
    if not capabilities:
        print(t("cli.capabilities.none", locale))
        return 0
    for capability in capabilities:
        status = capability.availability.value
        print(f"  {capability.id:<28} {status:<16} {capability.describe(locale)}")
    return 0


def _command_molecule(args: argparse.Namespace, locale: str) -> int:
    molecule = _load_molecule(args.path, charge=args.charge, multiplicity=args.multiplicity)
    print(t("cli.molecule.title", locale, name=molecule.name))
    print(f"  {t('cli.molecule.formula', locale, value=molecule.formula)}")
    print(f"  {t('cli.molecule.atoms', locale, value=molecule.n_atoms)}")
    print(f"  {t('cli.molecule.electrons', locale, value=molecule.n_electrons)}")
    state = t(
        "cli.molecule.state", locale, charge=molecule.charge, multiplicity=molecule.multiplicity
    )
    print(f"  {state}")
    issues = molecule.check_valence()
    if issues:
        print(f"  {t('cli.molecule.valence_issues', locale, count=len(issues))}")
        for issue in issues:
            print(
                "    "
                + t(
                    "editor.valence_warning",
                    locale,
                    symbol=issue.symbol,
                    index=issue.index,
                    actual=issue.observed,
                    expected=issue.expected,
                )
            )
    else:
        print(f"  {t('cli.molecule.valence_ok', locale)}")
    contacts = molecule.suspicious_contacts()
    print(f"  {t('cli.molecule.contacts', locale, count=len(contacts))}")
    return 0


def _resolution(
    args: argparse.Namespace, registry: CapabilityRegistry
) -> tuple[Molecule, CalculationSpec, Resolution | None]:
    """Строит молекулу и спецификацию: либо явную, либо по профилю автоподбора."""
    molecule = _load_molecule(args.path, charge=args.charge, multiplicity=args.multiplicity)
    spec = _build_spec(args, registry)
    hardware = HardwareContext(cores=args.cores, memory_mb=args.memory_mb, gpu_count=args.gpus)
    if spec.profile is not None:
        resolution = resolve_profile(
            spec.profile, task=spec.task, molecule=molecule, hardware=hardware
        )
        return molecule, resolution.spec, resolution
    return molecule, spec, None


def _command_plan(args: argparse.Namespace, registry: CapabilityRegistry, locale: str) -> int:
    _, spec, resolution = _resolution(args, registry)
    print(t("cli.run.header", locale))
    print(f"  {t('task.' + spec.task.value + '.title', locale)}")
    if resolution is not None:
        # Режим автоподбора: показываем каждое решение с обоснованием (§8 ТЗ).
        for line in resolution.explain(locale):
            print(f"  {line}")
        return 0
    # Экспертный режим: параметры заданы пользователем, объяснять нечего.
    if spec.method:
        print(f"  {t('profile.decision.method', locale, value=spec.method.theory.value)}")
        print(f"  {t('profile.decision.basis', locale, value=spec.method.basis)}")
        if spec.method.functional:
            print(f"  {t('profile.decision.functional', locale, value=spec.method.functional)}")
    return 0


def _command_run(args: argparse.Namespace, registry: CapabilityRegistry, locale: str) -> int:
    store = LocalJobStore(args.data_dir)
    molecule, spec, _ = _resolution(args, registry)

    job = Job(
        name=args.name or f"{molecule.name}-{spec.task.value}",
        project_id=args.project,
        owner=args.owner,
        spec=spec,
        molecule_uri=f"file://{store.molecule_path('pending')}",
        molecule_hash=molecule.structure_hash(),
        resources=spec.resources,
    )
    store.store_molecule(job.id, molecule.to_xyz())
    job = job.model_copy(update={"molecule_uri": f"file://{store.molecule_path(job.id)}"})
    store.save(job)
    print(t("cli.job.created", locale, id=job.id, name=job.name))

    store.update(job.id, lambda item: item.transition_to(JobStatus.QUEUED, actor="cli"))
    print(t("cli.job.queued", locale, id=job.id))

    engine = ReferenceEngine(registry)

    # Честная проверка доступности ДО запуска (§54 ТЗ): вместо имитации расчёта
    # сообщаем, чего именно не хватает. Задание остаётся в очереди — это правда:
    # оно будет выполнено, когда появится соответствующее ядро.
    try:
        basis_name = engine.assert_supported(spec)
    except QuantumLabError as error:
        status = store.load(job.id).status
        print(t("cli.run.engine_unavailable", locale, status=t(status.i18n_key, locale)))
        print(error.explain(locale))
        return 1

    method_label = spec.method.theory.value if spec.method else "—"
    print(
        t(
            "cli.run.started",
            locale,
            task=t(f"task.{spec.task.value}.title", locale),
            method=method_label,
            basis=basis_name,
        )
    )
    store.update(job.id, lambda item: item.transition_to(JobStatus.STARTING, actor="cli"))
    store.update(job.id, lambda item: item.transition_to(JobStatus.RUNNING, actor="cli"))

    try:
        result = engine.run(EngineRequest(job_id=job.id, molecule=molecule, spec=spec))
    except QuantumLabError as error:
        # Имя из «except ... as» удаляется по выходе из блока, поэтому диагноз
        # переводим в данные сразу: замыкание не должно ссылаться на переменную,
        # которой к моменту вызова уже не будет.
        code = str(error.code)
        params = {key: str(value) for key, value in error.params.items()}
        explanation = error.explain(locale)
        store.update(job.id, lambda item: _mark_failed(item, code, params))
        print(f"{t('cli.run.failed', locale)}: {explanation}", file=sys.stderr)
        return 1

    result_path = store.save_result(job.id, result.model_dump_json(indent=2))
    final = JobStatus.COMPLETED_WITH_WARNINGS if result.warnings else JobStatus.COMPLETED
    uri = f"file://{result_path}"
    store.update(job.id, lambda item: _mark_finished(item, uri, final))

    key = "cli.run.iterations" if result.converged else "cli.run.not_converged"
    print(f"  {t(key, locale, iterations=result.scf_iterations)}")
    print(
        "  "
        + t(
            "cli.run.summary",
            locale,
            energy=f"{result.energy_hartree:.10f}",
            homo=f"{result.homo_energy_hartree:.6f}",
            lumo=f"{result.lumo_energy_hartree:.6f}",
            dipole=f"{result.dipole_debye:.4f}",
        )
    )
    if result.warnings:
        print(f"  {t('cli.run.warnings', locale)}")
        for warning in result.warnings:
            print(f"    ! {warning}")
    print(f"  {t('cli.run.result_saved', locale, path=result_path)}")
    return 0 if result.converged else 1


def _mark_failed(job: Job, code: str, params: dict[str, str]) -> None:
    """Переводит задание в «не выполнено», сохраняя диагноз.

    Код и параметры ошибки сохраняются отдельно от текста: текст локализуется
    при показе, а код нужен машинам (повторы, отчёты, сопоставление с §26 ТЗ).
    """
    job.error_code = code
    job.error_params = params
    job.transition_to(JobStatus.FAILED, actor="cli")


def _mark_finished(job: Job, result_uri: str, status: JobStatus) -> None:
    """Привязывает результат к заданию и закрывает его."""
    job.result_uri = result_uri
    job.transition_to(status, actor="cli")


def _command_job(args: argparse.Namespace, locale: str) -> int:
    store = LocalJobStore(args.data_dir)
    command: str = args.job_command

    if command == "list":
        status = JobStatus(args.status) if args.status else None
        jobs = store.list(status)
        if not jobs:
            print(t("cli.job.empty", locale))
            return 0
        print(t("cli.job.list.header", locale))
        for job in jobs:
            status_text = t(job.status.i18n_key, locale)
            print(f"{job.id:<38} {status_text:<18} {job.attempt:^7}  {job.name}")
        return 0

    job_id: str = args.job_id
    try:
        if command == "status":
            job = store.load(job_id)
            print(f"{job.id}  {t(job.status.i18n_key, locale)}  {job.name}")
            print(f"  {t('task.' + job.spec.task.value + '.title', locale)}")
            print(f"  {t('summary.time', locale)}: {job.elapsed_seconds:.1f} c")
            return 0
        if command == "logs":
            job = store.load(job_id)
            print(t("cli.job.logs.title", locale, id=job.id))
            for event in job.events:
                previous = t(event.from_status.i18n_key, locale) if event.from_status else "—"
                target = t(event.to_status.i18n_key, locale)
                print(f"  {event.at.isoformat()}  {previous} → {target}  ({event.actor})")
            return 0
        if command == "cancel":
            store.update(job_id, lambda item: item.transition_to(JobStatus.CANCELLED, actor="cli"))
            print(t("cli.job.cancelled", locale, id=job_id))
            return 0
        if command == "resume":
            job = store.update(
                job_id, lambda item: item.transition_to(JobStatus.RUNNING, actor="cli")
            )
            print(t("cli.job.resumed", locale, id=job.id))
            return 0
        if command == "retry":
            job = store.update(job_id, lambda item: item.retry(actor="cli"))
            print(t("cli.job.retried", locale, id=job.id, attempt=job.attempt))
            return 0
    except LookupError:
        print(t("cli.job.not_found", locale, id=job_id, store=str(store.root)), file=sys.stderr)
        return 2
    except QuantumLabError as error:
        print(f"{t('cli.error.header', locale)}: {error.explain(locale)}", file=sys.stderr)
        return 2

    msg = f"Неизвестная подкоманда job {command!r}"
    raise AssertionError(msg)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
