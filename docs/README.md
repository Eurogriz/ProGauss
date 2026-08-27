# QuantumLab — архитектурный проект

**Рабочее название продукта:** QuantumLab
**Статус документа:** предложено к утверждению (Этап 1 — Foundation)
**Дата:** 2026-08-26
**Версия платформы:** 0.1.0 · контракт API `v1`

---

## О продукте одной фразой

> **Профессиональная квантовая химия, которой можно пользоваться без изучения
> языка input-файлов.** Сложность находится внутри системы, а не на плечах
> пользователя; при этом эксперт никогда не упирается в потолок интерфейса —
> полный контроль доступен через расширенные настройки, CLI и API.

## Что уже существует в репозитории

Документация описывает целевую архитектуру, но **не является обещанием**: слой
контрактов уже реализован, типизирован и покрыт тестами. Ниже — то, что можно
запустить прямо сейчас.

| Компонент | Статус | Где |
|---|---|---|
| Доменная модель (молекула, спецификация, задание, результат) | реализовано, 84 теста | `src/quantumlab/domain/` |
| Локализация ru (по умолчанию) + en, 311 ключей, паритет проверяется тестом | реализовано | `src/quantumlab/i18n/` |
| Человекочитаемые ошибки с блоками «что произошло / что попробовали / что сделать» | реализовано | `src/quantumlab/errors.py` |
| Контракты расчётного ядра (интегралы, SCF, DFT, корреляция, оптимизация, частоты, backend) | реализованы как Protocol | `src/quantumlab/engine/contracts.py` |
| Реестр возможностей (честные статусы «реализовано / частично / недоступно») | реализовано | `src/quantumlab/engine/registry.py` |
| Автоподбор параметров с объяснением каждого решения | реализовано | `src/quantumlab/recommend/profiles.py` |
| Машина состояний задания + атомарное файловое хранилище | реализовано | `src/quantumlab/jobs/`, `src/quantumlab/storage/` |
| Отпечаток воспроизводимости расчёта | реализовано | `src/quantumlab/domain/fingerprint.py` |
| CLI (`version`, `capabilities`, `molecule inspect`, `plan`, `run`, `job …`) | реализовано | `src/quantumlab/cli.py` |
| **Расчётное ядро (интегралы, SCF, DFT)** | **не реализовано** | — |

> **Важно (§54 ТЗ).** Ни один расчётный метод сейчас не объявлен доступным.
> `quantumlab capabilities` честно печатает `not_implemented`, а
> `quantumlab run` создаёт задание, ставит его в очередь и сообщает, что ядро
> не подключено — вместо того чтобы имитировать расчёт.

## Порядок чтения

1. [`architecture/01-overview.md`](architecture/01-overview.md) — архитектура, диаграмма, принципы
2. [`architecture/02-repository-structure.md`](architecture/02-repository-structure.md) — структура репозитория
3. [`architecture/03-technology-stack.md`](architecture/03-technology-stack.md) — стек и обоснование выбора
4. [`architecture/04-domain-model.md`](architecture/04-domain-model.md) — доменная модель и схема БД
5. [`architecture/05-api-contracts.md`](architecture/05-api-contracts.md) — REST API, CLI, Python SDK
6. [`architecture/06-quantum-engine.md`](architecture/06-quantum-engine.md) — расчётное ядро, алгоритмы, производительность
7. [`architecture/07-ux-flows.md`](architecture/07-ux-flows.md) — GUI, мастер, локализация, ошибки
8. [`architecture/08-execution-hpc.md`](architecture/08-execution-hpc.md) — выполнение, HPC, надёжность
9. [`architecture/09-verification-and-benchmark.md`](architecture/09-verification-and-benchmark.md) — верификация и бенчмарки
10. [`architecture/10-roadmap.md`](architecture/10-roadmap.md) — этапы, MVP, production roadmap
11. [`architecture/adr/`](architecture/adr/) — решения с обоснованием (ADR)

Практические руководства:

- [`user/quickstart.md`](user/quickstart.md) — «Мой первый расчёт за 5 минут» (рус.)
- [`dev/development.md`](dev/development.md) — инструкции разработчика
- [`ops/hpc-deployment.md`](ops/hpc-deployment.md) — развёртывание на кластере
