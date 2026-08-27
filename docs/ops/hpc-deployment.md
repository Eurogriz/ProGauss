# Развёртывание: от ноутбука до HPC-кластера

## 1. Режимы развёртывания

| Режим | Состав | Когда |
|---|---|---|
| **Локальный** | `quantumlab` CLI + файловое хранилище | один пользователь, ноутбук |
| **Single-node** | docker-compose: api, worker, postgres, redis, minio, grafana | рабочая группа |
| **Cluster** | Kubernetes + HPC-планировщик как исполнитель | производство |
| **HPC-only** | CLI/SDK на логин-узле, ядро через Slurm | классический кластер |

## 2. Локальный режим

```bash
pip install -e .
quantumlab run molecule.xyz --task optimize --profile standard
# состояние: ~/.quantumlab/{jobs,molecules,checkpoints}
```

Хранилище заданий пишет файлы атомарно (временный файл + `Path.replace`),
поэтому прерывание записи не оставляет повреждённое задание.

## 3. Single-node (docker-compose)

```yaml
# deploy/compose/single-node.yaml (появится в Этапе 3)
services:
  postgres:   { image: postgres:16, volumes: [pgdata:/var/lib/postgresql/data] }
  redis:      { image: redis:7 }
  minio:      { image: minio/minio, command: server /data }
  migrate:    { image: quantumlab-migrate, command: alembic upgrade head }
  api:        { image: quantumlab-api, ports: ["8000:8000"], depends_on: [migrate] }
  worker:     { image: quantumlab-worker, deploy: { replicas: 4 } }
  grafana:    { image: grafana/grafana }
```

```bash
docker compose -f deploy/compose/single-node.yaml up -d
curl -fsS localhost:8000/api/v1/ready
```

## 4. HPC-режим (§12 ТЗ)

На логин-узле кластера веб-сервис обычно не нужен: работает CLI/SDK, а задания
уходят в Slurm.

### 4.1 Настройка

```toml
# ~/.config/quantumlab/config.toml
[scheduler]
backend = "slurm"
partition = "compute"
account = "proj-1234"
qos = "normal"
time_limit = "24:00:00"

[parallelism]
model = "hybrid"          # MPI между узлами, OpenMP внутри узла
mpi_ranks_per_node = 2
threads_per_rank = 24     # = число физических ядер на сокет

[gpu]
enabled = true
gpus_per_node = 4
backend = "cuda"
```

### 4.2 Запуск

```bash
quantumlab run benzene.xyz --task optimize --profile high-accuracy \
    --nodes 2 --gpus 4 --scheduler slurm
```

Планировщик генерирует `sbatch`-скрипт с запросом ресурсов, привязкой потоков
и NUMA-политикой:

```bash
#SBATCH --job-name=quantumlab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00

srun --cpu-bind=cores --distribution=block:block \
     numactl --interleave=all \
     quantumlab-worker --job "$QLAB_JOB_ID"
```

### 4.3 Что видит пользователь

В разделе **HPC**: очередь, запущенные задания, CPU / RAM / GPU, время,
прогресс, ETA и ошибки — по каждому заданию видно узел и фактическое
потребление ресурсов (§12 ТЗ).

## 5. Kubernetes

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/postgres.yaml -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/api.yaml    -f deploy/k8s/worker.yaml
kubectl apply -f deploy/k8s/gpu-worker.yaml   # nodeSelector: nvidia.com/gpu
```

Ключевые моменты:

- `requests`/`limits` обязательны: задание без лимитов может убить узел;
- GPU-worker'ы — отдельный Deployment с `nvidia.com/gpu` и tolerations;
- `livenessProbe` → `/api/v1/health`, `readinessProbe` → `/api/v1/ready`;
- HorizontalPodAutoscaler для API; масштабирование worker'ов — по глубине
  очереди (`ql_queue_depth`).

## 6. Безопасность (§25 ТЗ)

| Требование | Реализация |
|---|---|
| Аутентификация | OIDC/JWT; локально — учётные записи |
| Авторизация | RBAC: `admin` / `researcher` / `viewer`; роли в проекте `owner` / `editor` / `viewer` |
| Изоляция проектов | все запросы фильтруются по членству в проекте; проверка в сервисном слое, а не в SQL-представлении |
| Секреты | внешний secret store (Vault/K8s Secrets), никогда не в репозитории |
| Транспорт | TLS везде; между компонентами — mTLS в кластере |
| Аудит | `audit_log` на каждое значимое действие |
| Пользовательские задания | контейнер без сети, read-only корень, `no-new-privileges`, лимиты CPU/RAM/PID, `seccomp` |
| Защита worker-узлов | выделенные очереди, отсутствие доступа к БД кроме API-токена задания |

**Произвольный пользовательский код не исполняется** (§25 ТЗ): экспертный
«сырой input» передаётся только адаптеру внешнего движка в sandbox'е и не
интерпретируется как скрипт.

## 7. Наблюдаемость (§33 ТЗ)

```bash
curl localhost:8000/api/v1/health     # liveness
curl localhost:8000/api/v1/ready      # БД, кэш, хранилище
curl localhost:8000/metrics           # Prometheus
```

Основные метрики:

| Метрика | Тип | Смысл |
|---|---|---|
| `ql_queue_depth` | gauge | глубина очереди — сигнал к масштабированию |
| `ql_job_duration_seconds{task,profile}` | histogram | время расчётов |
| `ql_scf_iterations` | histogram | сходимость SCF в среднем по задачам |
| `ql_integral_time_seconds` | histogram | доля времени в интегралах |
| `ql_scf_failures_total` | counter | тревожный рост = проблема с настройками |
| `ql_worker_busy` | gauge | загрузка исполнителей |
| `ql_gpu_utilization` | gauge | использование GPU |

Алерты: рост очереди выше порога, доля SCF-несходимостей, деградация времени на
эталонной молекуле (регрессия производительности), недоступность хранилища.

## 8. Резервное копирование и восстановление

- PostgreSQL: `pg_basebackup` + WAL-архив, восстановление на точку во времени;
- Object storage: версионирование bucket'ов + репликация;
- Регулярная тренировка восстановления: бекап, который ни разу не
  восстанавливали, бекапом не является.

## 9. Обновление версии

1. Применить миграции БД (`quantumlab-migrate`);
2. Обновить worker'ы (старые задания завершаются на старой версии — версия
   движка входит в отпечаток расчёта);
3. Обновить API;
4. Проверить `/ready` и эталонный расчёт из верификационного набора.

Обратная совместимость: в пределах `/api/v1` старые клиенты продолжают
работать; несовместимые изменения публикуются как `/api/v2` с сосуществованием
минимум одного релизного цикла.
