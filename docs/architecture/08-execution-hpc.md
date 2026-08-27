# 08. Выполнение, HPC, надёжность

## 8.1 Поток выполнения (§31 ТЗ)

```text
UI ──► API ──► Job Manager ──► Scheduler ──► Worker ──► Quantum Engine ──► Result Store
```

Каждая стрелка — явная граница с контрактом:

| Граница | Контракт | Что не пересекает границу |
|---|---|---|
| UI → API | OpenAPI `/api/v1` | доменные объекты pydantic |
| API → Job Manager | `Job`, `CalculationSpec` | HTTP-контекст, сессия |
| Job Manager → Scheduler | очередь в PostgreSQL + `ResourceSpec` | детали расчёта |
| Scheduler → Worker | манифест задания (JSON) + чекпоинт-URI | состояние API-сервера |
| Worker → Engine | `EngineRequest` → `CalculationResult` | очередь, пользователь, хранилище |
| Engine → Store | скаляры в SQL, массивы как артефакты | временные файлы |

## 8.2 Job Manager (§13 ТЗ)

Обязанности:

1. Создание задания: UUID, имя, владелец, проект, спецификация, отпечаток,
   ссылка на структуру, `molecule_hash`;
2. Атомарные переходы состояний с журналом (`job_events`);
3. Retry-политика: экспоненциальная задержка, максимум N попыток, номер
   попытки входит в ключ чекпоинта (идемпотентность);
4. Таймауты: wall-time и «нет прогресса X минут»;
5. Отмена: кооперативная — worker получает сигнал и корректно завершает шаг,
   сохраняя чекпоинт;
6. Аудит: кто и что сделал с заданием.

**Выборка из очереди** — один SQL-запрос без гонок:

```sql
UPDATE jobs SET status = 'starting', worker_id = $1, started_at = now()
WHERE id = (
  SELECT id FROM jobs
  WHERE status = 'queued'
    AND project_id = ANY($2)          -- изоляция проектов
  ORDER BY priority DESC, created_at
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
RETURNING *;
```

`SKIP LOCKED` даёт корректную конкуренцию нескольких планировщиков без
внешней очереди; частичный индекс `jobs_queue_idx` делает выборку дешёвой при
миллионах завершённых заданий.

## 8.3 Планировщики (§12 ТЗ)

```python
class SchedulerAdapter(Protocol):
    def submit(self, manifest: JobManifest) -> ExternalJobId: ...
    def poll(self, external_id: ExternalJobId) -> ExternalState: ...
    def cancel(self, external_id: ExternalJobId) -> None: ...
    def resources(self) -> ClusterResources: ...
```

| Планировщик | Особенности |
|---|---|
| `local` | пул процессов на одной машине; режим по умолчанию для ноутбука |
| `slurm` | приоритет: `sbatch` с шаблоном, учёт partition/account/QOS, `scontrol` для статусов |
| `pbs` | `qsub`/`qstat`, маппинг состояний |
| `lsf` | `bsub`/`bjobs` |
| `kubernetes` | Job-объекты с запросами/лимитами ресурсов, GPU через `nvidia.com/gpu` |

Гибридная модель: **MPI между узлами, OpenMP внутри узла, GPU — на узел**.
Количество потоков и рангов вычисляется планировщиком из `ResourceSpec` и
реальной топологии (сокеты, NUMA-узлы, GPU на узел), а не задаётся
пользователем вручную.

Пользователь в разделе **HPC** видит: очередь, запущенные задания, CPU, RAM,
GPU, время, прогресс, ETA, ошибки — и по каждому заданию: на каком узле оно
идёт и сколько ресурсов реально потребляет.

## 8.4 Надёжность и восстановление (§14 ТЗ)

| Отказ | Механизм | Результат для пользователя |
|---|---|---|
| Падение worker-процесса | heartbeat в Redis + чекпоинт; Job Manager замечает пропажу и переводит задание в `failed` → автоматически в `queued` | «Расчёт прерван. Продолжаем с контрольной точки» |
| Нехватка памяти | диагностика `runtime.out_of_memory` с предложениями | Кнопки: увеличить память / включить direct SCF / уменьшить базис |
| Отключение worker'а | идемпотентный повтор: та же `(job_id, attempt)` не создаёт дубль артефактов | Задание продолжается, история не дублируется |
| Потеря сети | экспоненциальный retry отчётов прогресса; прогресс кэшируется локально | Прогресс «догоняет» после восстановления |
| Перезапуск сервера | состояние — в SQL, не в памяти процесса | Очередь продолжается с того же места |
| Частичная потеря узла | MPI-ранги рестартуются планировщиком; при невозможности — рестарт с чекпоинта на другом узле | Задание завершается, пусть и позже |
| Повреждённый артефакт | `sha256` проверяется при чтении | Явная ошибка `storage.artifact_missing`, а не молча неверные числа |

### Атомарность и идемпотентность

- Переход состояния — одна SQL-транзакция (`UPDATE jobs` + `INSERT job_events`);
- Запись файла/артефакта — во временный объект и атомарный `rename`
  (в локальном хранилище реализовано через `Path.replace`, см.
  `quantumlab/storage/local_jobs.py`);
- Ключ артефакта content-addressed: повторная запись того же массива ничего не
  меняет;
- Повтор задания увеличивает `attempt`, поэтому чекпоинты попыток не
  перезаписывают друг друга.

### Кнопка «Продолжить расчёт»

`POST /jobs/{id}/resume` → `Job.ensure_resumable()`:

- есть `checkpoint_uri`;
- `molecule_hash` и `spec_sha` совпадают с сохранёнными (иначе — ошибка,
  потому что продолжение расчёта с другой геометрией бессмысленно);
- статус допускает возобновление.

При отсутствии чекпоинта пользователь получает `job.not_resumable` с
объяснением и кнопкой «Запустить заново» — а не пустую ошибку.

## 8.5 Пакетный режим (§38 ТЗ)

```text
1000 молекул ──► генерация заданий ──► очередь ──► параллельное выполнение
                                                      │
                     CSV / JSON / Parquet ◄── агрегация результатов
```

- **Concurrency control**: лимит одновременных заданий на проект и глобально;
- **Retries**: на задание, с сохранением причины каждой неудачи;
- **Checkpoint**: состояние пакета — отдельная сущность, прерванный пакет
  продолжается, а не начинается заново;
- **Failed-job handling**: ошибка одного расчёта не останавливает пакет;
  в экспорте — колонка `error_code`;
- **Агрегация**: энергии, gaps, диполи — одной таблицей, экспорт в Parquet.

## 8.6 Workflow (§37 ТЗ)

Пайплайн описывается декларативно (без программирования):

```yaml
name: Полный цикл для бензола
steps:
  - id: preopt
    task: optimization
    profile: screening
  - id: opt
    task: optimization
    profile: standard
    input: { from: preopt, geometry: final }
  - id: freq
    task: frequencies
    input: { from: opt, geometry: final }
  - id: props
    task: properties
    input: { from: opt, geometry: final }
    options: { orbitals: [homo, lumo], esp: true }
  - id: report
    task: report
    input: { from: [opt, freq, props] }
```

Шаги образуют DAG; планировщик запускает независимые шаги параллельно. Отказ
шага останавливает зависимые и предлагает варианты: пропустить, повторить,
изменить параметры.

## 8.7 Развёртывание (§32 ТЗ)

| Режим | Состав | Назначение |
|---|---|---|
| **Локальный** | `quantumlab serve` + SQLite/файловое хранилище | ноутбук, один пользователь |
| **Single-node** | docker-compose: api, worker, postgres, redis, minio, grafana | рабочая группа |
| **Cluster** | те же образы в Kubernetes + HPC-планировщик как исполнитель | производство |
| **HPC-only** | CLI + SDK на логин-узле, ядро через Slurm | классический кластер без веб-сервиса |

Образы: `quantumlab-api`, `quantumlab-worker`, `quantumlab-worker-gpu`
(CUDA-базовый образ), `quantumlab-migrate` (одноразовый job для миграций).

Подробности — в [`../ops/hpc-deployment.md`](../ops/hpc-deployment.md).

## 8.8 Наблюдаемость (§33 ТЗ)

| Что | Как |
|---|---|
| Структурные логи | JSON: `job_id`, `attempt`, `stage`, `iteration`, `worker`, `event` |
| Метрики | Prometheus: `ql_job_duration_seconds{task,profile}`, `ql_scf_iterations`, `ql_integral_time_seconds`, `ql_queue_depth`, `ql_worker_busy`, `ql_gpu_utilization` |
| Трейсинг | OpenTelemetry: span на каждый этап (integrals → scf → gradient → …) |
| Health | `GET /health` (liveness), `GET /ready` (БД, кэш, хранилище) |
| Job metrics | прогресс, ETA, потребление CPU/RAM/GPU по каждому заданию |
| Алерты | SCF-несходимости выше порога, рост очереди, деградация времени на эталонной молекуле |

Последний пункт принципиален: **регрессия производительности обнаруживается по
эталонной молекуле**, а не по жалобе пользователя.
