"""Сборка FastAPI-приложения по контракту ``api/openapi/v1.yaml``.

Приложение создаётся фабрикой :func:`create_app`, чтобы тесты поднимали его
на временном каталоге данных и не зависели от глобального состояния.

Что реализовано и что нет — см. ``README.md`` и ``/capabilities``. Каждый
нереализованный путь контракта отдаёт ``501`` с локализованным ``Problem``.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from quantumlab.domain.job import Job
from quantumlab.domain.result import CalculationResult
from quantumlab.domain.spec import CalculationSpec, PrecisionProfile, Task
from quantumlab.engine.registry import default_registry
from quantumlab.errors import CatalogEntryNotFoundError, QuantumLabError
from quantumlab.i18n import DEFAULT_LOCALE, SUPPORTED_LOCALES, get_catalog, t
from quantumlab.jobs.state_machine import JobStatus
from quantumlab.recommend.profiles import HardwareContext, resolve_profile
from quantumlab.storage.local_catalog import LocalCatalog, MoleculeRecord
from quantumlab.storage.local_jobs import LocalJobStore
from quantumlab.version import __version__

#: JSON-значение — то, что может лежать в теле ответа. Алиас рекурсивный и
#: потому инвариантный: на входе :func:`to_api` принимает ``object``.
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

#: Асинхронная заглушка нереализованной операции.
StubHandler = Callable[[str | None], Coroutine[Any, Any, JSONResponse]]

#: Код ошибки → HTTP-статус. Всё остальное из таксономии — 422.
_STATUS_BY_CODE: dict[str, int] = {
    "catalog.not_found": 404,
    "registry.basis_not_found": 404,
    "registry.functional_not_found": 404,
    "engine.method_not_available": 501,
    "format.not_implemented": 501,
    "storage.artifact_missing": 404,
    "job.not_resumable": 409,
    "job.invalid_transition": 409,
}

#: Пути контракта без реализации. Отдаются как 501, а не прячутся.
UNIMPLEMENTED_OPERATIONS: dict[str, tuple[str, str]] = {
    "POST /auth/token": ("/auth/token", "post"),
    "GET /auth/me": ("/auth/me", "get"),
    "POST /jobs/{jobId}/cancel": ("/jobs/{job_id}/cancel", "post"),
    "POST /jobs/{jobId}/resume": ("/jobs/{job_id}/resume", "post"),
    "POST /jobs/{jobId}/retry": ("/jobs/{job_id}/retry", "post"),
    "GET /jobs/{jobId}/events": ("/jobs/{job_id}/events", "get"),
    "GET /jobs/{jobId}/logs": ("/jobs/{job_id}/logs", "get"),
    "GET /jobs/{jobId}/report": ("/jobs/{job_id}/report", "get"),
    "POST /jobs/{jobId}/result/export": ("/jobs/{job_id}/result/export", "post"),
    "GET /artifacts/{artifactId}": ("/artifacts/{artifact_id}", "get"),
    "GET /workers": ("/workers", "get"),
}

LanguageHeader = Annotated[str | None, Header(alias="Accept-Language")]


# --------------------------------------------------------------------------- #
# Приведение к соглашениям контракта
# --------------------------------------------------------------------------- #
def _camel(key: str) -> str:
    """``snake_case`` → ``camelCase`` для одного ключа."""
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def to_api(payload: object) -> JsonValue:
    """Приводит ответ к соглашениям контракта: ключи в ``camelCase``.

    Доменная модель остаётся в ``snake_case`` — от неё зависят отпечаток
    расчёта и JSON артефактов на диске, и менять их ради формы ответов нельзя.
    Преобразование делается здесь, на границе HTTP, и рекурсивно: вложенные
    структуры (``finalMolecule.atoms``, ``qualityChecks``) обязаны
    соответствовать схеме так же, как верхний уровень.
    """
    if isinstance(payload, dict):
        return {_camel(str(key)): to_api(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [to_api(item) for item in payload]
    return cast("JsonValue", payload)


def _snake(key: str) -> str:
    """``camelCase`` → ``snake_case`` для одного ключа."""
    out: list[str] = []
    for index, char in enumerate(key):
        if char.isupper() and index > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def from_api(payload: object) -> object:
    """Обратное к :func:`to_api`: тело запроса в ``camelCase`` → домен.

    Без этого поток «получил план — отправил расчёт» разваливался: план
    приходит в camelCase, а доменная модель принимает только snake_case.
    Клиент не должен знать, в какой нотации устроены внутренние модели.
    """
    if isinstance(payload, dict):
        return {_snake(str(key)): from_api(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [from_api(item) for item in payload]
    return payload


def problem(status: int, code: str, error: QuantumLabError, locale: str) -> JSONResponse:
    """Собирает RFC 9457 ``Problem`` из доменной ошибки."""
    body: dict[str, JsonValue] = {
        "type": f"https://quantumlab.dev/errors/{code}",
        "title": error.title(locale),
        "status": status,
        "detail": error.what_happened(locale),
        "code": code,
    }
    hint = error.hint(locale)
    if hint:
        body["hint"] = hint
    return JSONResponse(status_code=status, content=body)


def locale_from(accept_language: str | None) -> str:
    """Разбирает ``Accept-Language`` до поддерживаемой локали.

    Берётся первое совпадение по языку; неизвестный язык откатывается к
    русскому, потому что он в интерфейсе главный (§3 ТЗ).
    """
    if not accept_language:
        return DEFAULT_LOCALE
    for chunk in accept_language.split(","):
        code = chunk.split(";")[0].strip().lower()
        if code in SUPPORTED_LOCALES:
            return code
        if code.split("-")[0] in SUPPORTED_LOCALES:
            return code.split("-")[0]
    return DEFAULT_LOCALE


# --------------------------------------------------------------------------- #
# Тела запросов
# --------------------------------------------------------------------------- #
class ApiModel(BaseModel):
    """База тел запросов: принимает ключи и в ``camelCase``, и в ``snake_case``.

    Контракт объявлен в ``camelCase``, а доменные модели живут в ``snake_case``;
    ``populate_by_name`` оставляет рабочие и те тела, что собраны вручную из
    доменных объектов (CLI, тесты).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, alias_generator=_camel)


class ProjectCreateRequest(ApiModel):
    """Тело ``POST /projects``."""

    name: str = Field(min_length=1, max_length=200)


class MoleculeCreateRequest(ApiModel):
    """Тело ``POST /projects/{projectId}/molecules``."""

    name: str | None = None
    charge: int = 0
    multiplicity: int = 1
    format: str = "xyz"
    content: str


class PlanRequestBody(ApiModel):
    """Тело ``POST /calculations/plan``."""

    task: Task
    profile: PrecisionProfile
    molecule_id: str
    hardware: dict[str, int] | None = None


class JobCreateRequest(ApiModel):
    """Тело ``POST /jobs``."""

    name: str | None = None
    molecule_id: str
    spec: CalculationSpec
    priority: int = Field(default=100, ge=0, le=1000)
    tags: tuple[str, ...] = ()

    @field_validator("spec", mode="before")
    @classmethod
    def _spec_from_api(cls, value: object) -> object:
        """Принимает спецификацию в camelCase: план приходит именно в ней."""
        return from_api(value)


# --------------------------------------------------------------------------- #
# Зависимости
# --------------------------------------------------------------------------- #
class Services:
    """Общее состояние приложения: хранилища, реестр, ядро."""

    def __init__(self, data_dir: Path) -> None:
        """Создаёт сервисы поверх каталога данных ``data_dir``."""
        self.data_dir = data_dir
        self.catalog = LocalCatalog(data_dir / "catalog")
        self.jobs = LocalJobStore(data_dir / "jobs")
        self.registry = default_registry()


def get_services(request: Request) -> Services:
    """Возвращает сервисы из состояния приложения."""
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


# --------------------------------------------------------------------------- #
# Фабрика приложения
# --------------------------------------------------------------------------- #
def create_app(data_dir: Path | str = "data") -> FastAPI:
    """Создаёт приложение поверх каталога данных ``data_dir``."""
    app = FastAPI(
        title="QuantumLab API",
        version=__version__,
        description="REST API платформы QuantumLab (контракт: api/openapi/v1.yaml)",
    )
    app.state.services = Services(Path(data_dir))

    @app.exception_handler(QuantumLabError)
    async def _quantumlab_error(request: Request, error: QuantumLabError) -> JSONResponse:
        locale = locale_from(request.headers.get("accept-language"))
        return problem(_STATUS_BY_CODE.get(error.code.value, 422), error.code.value, error, locale)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        """Отдаёт ``Problem`` вместо дефолтного ``{"detail": [...]}``.

        FastAPI бросает ``RequestValidationError``, а не pydantic-овский
        ``ValidationError``: обработчик последнего сюда не попадает никогда, и
        клиент получил бы тело без ``code``, ``type`` и локализованного заголовка.
        """
        locale = locale_from(request.headers.get("accept-language"))
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://quantumlab.dev/errors/validation.failed",
                "title": t("error.validation.title", locale),
                "status": 422,
                "code": "validation.failed",
                "detail": str(error),
            },
        )

    _register_health(app)
    _register_catalog_routes(app)
    _register_calculation_routes(app)
    _register_job_routes(app)
    _register_stubs(app)
    return app


# --------------------------------------------------------------------------- #
# Служебные и справочные эндпоинты
# --------------------------------------------------------------------------- #
def _register_health(app: FastAPI) -> None:
    @app.get("/health", tags=["service"], response_model=None)
    async def health() -> JsonValue:
        """Жив ли процесс. Проверок не выполняет."""
        return {"status": "ok"}

    @app.get("/ready", tags=["service"], response_model=None)
    async def ready(services: ServicesDep) -> JsonValue:
        """Готов ли принимать расчёты: хранилища читаются."""
        checks: dict[str, bool] = {}
        try:
            services.jobs.list()
            services.catalog.list_projects()
            checks["storage"] = True
        except OSError:
            checks["storage"] = False
        status = "ready" if all(checks.values()) else "not_ready"
        return to_api({"status": status, "checks": checks})

    @app.get("/capabilities", tags=["service"], response_model=None)
    async def capabilities(services: ServicesDep) -> JsonValue:
        """Срез реестра возможностей — те же данные, что в ``quantumlab capabilities``."""
        return to_api(services.registry.snapshot())

    @app.get("/i18n/{locale}", tags=["service"], response_model=None)
    async def i18n_catalog(locale: str) -> JsonValue:
        """Каталог строк интерфейса; клиент не хранит текстов в коде (§3 ТЗ)."""
        if locale not in SUPPORTED_LOCALES:
            raise HTTPException(status_code=404, detail=f"locale {locale} is not supported")
        return to_api(get_catalog(locale).messages())


# --------------------------------------------------------------------------- #
# Проекты и структуры
# --------------------------------------------------------------------------- #
def _register_catalog_routes(app: FastAPI) -> None:
    @app.post("/projects", status_code=201, tags=["catalog"], response_model=None)
    async def create_project(body: ProjectCreateRequest, services: ServicesDep) -> JsonValue:
        """Создаёт проект — границу изоляции структур и заданий."""
        record = services.catalog.create_project(body.name)
        return to_api(
            {
                "id": record.id,
                "name": record.name,
                "role": record.role,
                "created_at": record.created_at.isoformat(),
            }
        )

    @app.get("/projects", tags=["catalog"], response_model=None)
    async def list_projects(services: ServicesDep) -> JsonValue:
        """Список проектов."""
        items = [
            {
                "id": item.id,
                "name": item.name,
                "role": item.role,
                "created_at": item.created_at.isoformat(),
            }
            for item in services.catalog.list_projects()
        ]
        return to_api({"items": items, "total": len(items)})

    @app.post(
        "/projects/{project_id}/molecules", status_code=201, tags=["catalog"], response_model=None
    )
    async def create_molecule(
        project_id: str, body: MoleculeCreateRequest, services: ServicesDep
    ) -> JsonValue:
        """Разбирает и сохраняет структуру в проекте."""
        record = services.catalog.create_molecule(
            project_id=project_id,
            name=body.name,
            content=body.content,
            fmt=body.format,
            charge=body.charge,
            multiplicity=body.multiplicity,
        )
        return to_api(_molecule_view(record))

    @app.get("/projects/{project_id}/molecules", tags=["catalog"], response_model=None)
    async def list_molecules(project_id: str, services: ServicesDep) -> JsonValue:
        """Структуры проекта."""
        services.catalog.get_project(project_id)
        items = [_molecule_view(m) for m in services.catalog.list_molecules(project_id)]
        return to_api({"items": items})

    @app.get("/molecules/{molecule_id}", tags=["catalog"], response_model=None)
    async def get_molecule(molecule_id: str, services: ServicesDep) -> JsonValue:
        """Структура по идентификатору."""
        return to_api(_molecule_view(services.catalog.get_molecule(molecule_id)))

    @app.post("/molecules/{molecule_id}/validate", tags=["catalog"], response_model=None)
    async def validate_molecule(
        molecule_id: str, services: ServicesDep, accept_language: LanguageHeader = None
    ) -> JsonValue:
        """Проверяет сохранённую структуру.

        Разбор выполняется при сохранении, поэтому сюда доходят только
        принятые структуры: отчёт подтверждает, что структура прошла доменные
        инварианты, и сообщает число электронов — по нему видно, годится ли
        расчёт RHF (нечётное число электронов требует UHF/ROHF).
        """
        locale = locale_from(accept_language)
        record = services.catalog.get_molecule(molecule_id)
        electrons = record.molecule.n_electrons
        issues: list[JsonValue] = []
        if electrons % 2 != 0:
            issues.append(
                {
                    "code": "molecule.invalid_multiplicity",
                    "message": t(
                        "error.molecule.invalid_multiplicity.what",
                        locale,
                        charge=record.charge,
                        electrons=electrons,
                        multiplicity=record.multiplicity,
                    ),
                }
            )
        return to_api({"valid": not issues, "issues": issues, "suggestions": []})


def _molecule_view(record: MoleculeRecord) -> dict[str, JsonValue]:
    return {
        "id": record.id,
        "name": record.name,
        "charge": record.charge,
        "multiplicity": record.multiplicity,
        "atoms": [
            {"symbol": atom.symbol, "position": list(atom.position)}
            for atom in record.molecule.atoms
        ],
    }


# --------------------------------------------------------------------------- #
# Подбор параметров
# --------------------------------------------------------------------------- #
def _register_calculation_routes(app: FastAPI) -> None:
    @app.post("/calculations/plan", tags=["calculations"], response_model=None)
    async def plan(
        body: PlanRequestBody, services: ServicesDep, accept_language: LanguageHeader = None
    ) -> JsonValue:
        """Разворачивает профиль точности в спецификацию с обоснованиями (§8 ТЗ)."""
        locale = locale_from(accept_language)
        record = services.catalog.get_molecule(body.molecule_id)
        hardware = body.hardware or {}
        resolution = resolve_profile(
            body.profile,
            task=body.task,
            molecule=record.molecule,
            hardware=HardwareContext(
                cores=hardware.get("cores", 1),
                memory_mb=hardware.get("memoryMb", 4096),
                gpu_count=hardware.get("gpuCount", 0),
            ),
        )
        return to_api(
            {
                "spec": _spec_view(resolution.spec),
                "decisions": [
                    {
                        "parameter": decision.parameter,
                        "value": decision.value,
                        "text": decision.render(locale),
                    }
                    for decision in resolution.decisions
                ],
                "rationale": resolution.explain(locale)[0],
            }
        )


# --------------------------------------------------------------------------- #
# Задания
# --------------------------------------------------------------------------- #
def _register_job_routes(app: FastAPI) -> None:
    @app.post("/jobs", status_code=202, tags=["calculations"], response_model=None)
    async def submit_job(body: JobCreateRequest, services: ServicesDep) -> JsonValue:
        """Принимает расчёт в очередь. Выполняет его воркер, не HTTP-запрос."""
        record = services.catalog.get_molecule(body.molecule_id)
        services.registry.assert_available(f"task:{body.spec.task.value}")
        job = Job(
            name=body.name or f"{record.name}-{body.spec.task.value}",
            project_id=record.project_id,
            owner="api",
            spec=body.spec,
            molecule_uri=f"molecule://{record.id}",
            molecule_hash=record.molecule.structure_hash(),
            resources=body.spec.resources,
            priority=body.priority,
        )
        # Контракт обещает 202 «принято в очередь», поэтому задание сразу
        # переводится из DRAFT в QUEUED: оставить его черновиком значило бы
        # принять расчёт и не поставить его в очередь.
        job.transition_to(JobStatus.QUEUED, actor="api")
        services.jobs.save(job)
        return to_api(_job_view(job))

    @app.get("/jobs", tags=["calculations"], response_model=None)
    async def list_jobs(
        services: ServicesDep,
        status: JobStatus | None = None,
        project_id: Annotated[str | None, Query(alias="projectId")] = None,
    ) -> JsonValue:
        """Очередь и история заданий."""
        items = [job for job in services.jobs.list(status) if _matches(job, project_id)]
        return to_api({"items": [_job_view(job) for job in items], "total": len(items)})

    @app.get("/jobs/{job_id}", tags=["calculations"], response_model=None)
    async def get_job(job_id: str, services: ServicesDep) -> JsonValue:
        """Состояние задания."""
        return to_api(_job_view(_load_job(services, job_id)))

    @app.get("/jobs/{job_id}/result", tags=["calculations"], response_model=None)
    async def get_result(job_id: str, services: ServicesDep) -> JsonValue:
        """Результат выполненного задания."""
        job = _load_job(services, job_id)
        path = services.jobs.result_path(job.id)
        if not path.exists():
            raise HTTPException(status_code=409, detail="result is not available yet")
        result = CalculationResult.model_validate_json(path.read_text(encoding="utf-8"))
        return to_api(result.model_dump(mode="json"))


def _matches(job: Job, project_id: str | None) -> bool:
    return project_id is None or job.project_id == project_id


def _load_job(services: Services, job_id: str) -> Job:
    try:
        return services.jobs.load(job_id)
    except LookupError as error:
        # LocalJobStore.load бросает LookupError; KeyError — его подкласс,
        # а не наоборот, поэтому перехватывать нужно именно надкласс.
        raise CatalogEntryNotFoundError("job", job_id) from error


def _job_view(job: Job) -> dict[str, JsonValue]:
    return {
        "id": job.id,
        "name": job.name,
        "project_id": job.project_id,
        "status": job.status.value,
        "attempt": job.attempt,
        "priority": job.priority,
        "created_at": job.created_at.isoformat(),
        "spec": _spec_view(job.spec),
    }


def _spec_view(spec: CalculationSpec) -> JsonValue:
    return cast("JsonValue", spec.model_dump(mode="json", by_alias=True))


# --------------------------------------------------------------------------- #
# Честные заглушки
# --------------------------------------------------------------------------- #
def _register_stubs(app: FastAPI) -> None:
    """Регистрирует нереализованные пути контракта как 501.

    Без этого клиент получил бы 404 и решил бы, что пути нет в API, хотя
    контракт его объявляет. 501 с кодом ``api.not_implemented`` говорит правду:
    операция заявлена, но ещё не написана.
    """

    def make_stub(operation: str) -> StubHandler:
        async def stub(accept_language: LanguageHeader = None) -> JSONResponse:
            locale = locale_from(accept_language)
            return JSONResponse(
                status_code=501,
                content={
                    "type": "https://quantumlab.dev/errors/api.not_implemented",
                    "title": t("error.api.not_implemented.title", locale),
                    "status": 501,
                    "code": "api.not_implemented",
                    "detail": t("error.api.not_implemented.what", locale, operation=operation),
                    "hint": t("error.api.not_implemented.hint", locale),
                },
            )

        slug = operation.replace(" ", "_").replace("/", "_").replace("{", "").replace("}", "")
        stub.__name__ = f"stub_{slug}"
        stub.__doc__ = f"Не реализовано: {operation}."
        return stub

    for operation, (path, method) in UNIMPLEMENTED_OPERATIONS.items():
        app.add_api_route(
            path,
            make_stub(operation),
            methods=[method.upper()],
            tags=["not-implemented"],
            summary=f"Не реализовано: {operation}",
        )
