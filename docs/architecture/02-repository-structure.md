# 02. Структура репозитория

Монорепозиторий: платформа, ядро и UI живут вместе, потому что изменение
контракта (`CalculationSpec`, `CalculationResult`) почти всегда затрагивает все
слои, и разъехавшиеся версии — главный источник production-инцидентов в
научном ПО. Границы внутри монорепо жёсткие: каждый пакет имеет собственную
зону ответственности и не импортирует «наискосок».

```text
ProGauss/
├── pyproject.toml              # упаковка, strict mypy, ruff, pytest-маркеры
├── Makefile                    # единые точки входа: make check / test / bench
├── README.md
│
├── src/quantumlab/             # ★ платформенный слой (Python, strict typing)
│   ├── domain/                 #   доменные модели — общее ядро языка
│   │   ├── molecule.py         #     атомы, связи, заряд/мультиплетность, проверки
│   │   ├── spec.py             #     CalculationSpec: задача, метод, SCF, сетка, ресурсы
│   │   ├── job.py              #     Job: UUID, статусы, события, чекпоинт
│   │   ├── result.py           #     CalculationResult + ссылки на артефакты
│   │   └── fingerprint.py      #     отпечаток воспроизводимости (§40 ТЗ)
│   ├── engine/
│   │   ├── contracts.py        #   Protocol-интерфейсы ядра и backend'ов
│   │   ├── capabilities.py     #   Availability / Capability
│   │   └── registry.py         #   реестр возможностей (единственный источник правды)
│   ├── recommend/profiles.py   #   автоподбор параметров + обоснования (§8 ТЗ)
│   ├── jobs/state_machine.py   #   допустимые переходы состояний (§13 ТЗ)
│   ├── storage/local_jobs.py   #   атомарное локальное хранилище заданий
│   ├── i18n/                   #   локализация (§3 ТЗ)
│   │   ├── catalog.py
│   │   └── locales/{ru,en}.json
│   ├── errors.py               #   таксономия ошибок и диагнозов (§19 ТЗ)
│   ├── cli.py                  #   CLI (§20 ТЗ)
│   └── version.py
│
├── native/                     # Этап 2+: C++20 ядро (см. 06-quantum-engine.md)
│   ├── integrals/              #   Obara–Saika / McMurchie–Davidson, SIMD, screening
│   ├── scf/                    #   DIIS/EDIIS, damping, level shifting, direct SCF
│   ├── dft/                    #   квадратура, XC-ядра
│   ├── correlation/            #   MP2 → CCSD → CCSD(T)
│   ├── backends/{cpu,cuda,rocm}/
│   └── bindings/               #   nanobind-обвязка, реализующая Protocol'ы
│
├── server/                     # Этап 3+: api-server, job-manager, scheduler, workers
│   ├── api/                    #   маршруты /api/v1 (тонкие, без бизнес-логики)
│   ├── jobmanager/             #   очередь, retry-политики, аудит
│   ├── scheduler/{local,slurm,pbs,lsf,k8s}/
│   ├── worker/                 #   runtime worker'а: sandbox, лимиты, чекпоинты
│   └── migrations/             #   Alembic-миграции схемы БД
│
├── web/                        # Этап 3+: Web UI (React + TS + Vite)
│   ├── src/features/{molecules,calculations,queue,results,visualization}/
│   ├── src/widgets/molecule-viewer/   # WebGL: атомы, связи, изоповерхности, МО
│   └── src/i18n/                      # только ключи; строки — из каталога платформы
│
├── api/openapi/v1.yaml         # контракт REST API (проверяется тестом)
├── plugins/                    # пример плагинов (§41 ТЗ): адаптеры движков, форматы
├── benchmarks/                 # воспроизводимые бенчмарки (§27 ТЗ)
│   ├── suites/{small,medium,large}/
│   └── reference/              # эталонные конфигурации и пороги деградации
├── verification/               # научная верификация (§26 ТЗ)
│   ├── datasets/               # эталонные энергии, геометрии, частоты
│   └── cases/                  # сценарии: молекула → ожидаемые значения
├── docs/                       # документация (этот каталог)
├── deploy/
│   ├── docker/{api,worker,worker-gpu}/Dockerfile
│   ├── compose/                # локальный и прод-профиль
│   ├── k8s/                    # манифесты
│   └── monitoring/             # Prometheus/Grafana/OTel
├── tests/                      # unit + интеграционные (§47 ТЗ)
└── .github/workflows/ci.yml    # lint → unit → integration → scientific → perf → security → build
```

## Правила зависимостей

```text
web, cli, sdk, api  ──►  quantumlab (domain, recommend, registry, i18n, jobs)
server              ──►  quantumlab
quantumlab          ──►  (ничего из server/web)
native              ──►  реализует Protocol'ы из quantumlab.engine.contracts
```

Нарушение направления проверяется в CI (import-linter в Этапе 3).

## Что намеренно НЕ в Git

- `*.chk`, `*.wfn`, `*.cube`, `*.npz`, `scratch/`, `work/` — расчётные артефакты;
- данные бенчмарков крупнее нескольких МБ — хранятся в object storage / LFS,
  в репозитории лежат только их манифесты и контрольные суммы;
- `.venv`, кэши инструментов.

Это не косметика: один кубический файл электронной плотности крупной молекулы —
десятки МБ, и попав в историю Git он остаётся там навсегда.
