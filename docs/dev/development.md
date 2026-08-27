# Инструкции разработчика

## 1. Окружение

```bash
git clone <репозиторий> && cd ProGauss
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make check            # lint + форматирование + mypy + тесты
```

Требуется Python ≥ 3.11. Нативная часть (Этап 2+) потребует CMake, C++20-компилятор
и BLAS/LAPACK.

## 2. Команды

```bash
make lint          # ruff check
make fmt           # ruff format
make typecheck     # mypy --strict
make test          # pytest
make bench         # бенчмарки (появятся в Этапе 4)
make check         # всё сразу — то, что запускает CI
```

## 3. Стандарты кода (§47 ТЗ)

| Правило | Инструмент | Замечание |
|---|---|---|
| Строгая типизация | `mypy --strict` | `Any` запрещён; исключения — с `# noqa: ANN401` и объяснением |
| Линт | `ruff check` | `RUF001/002/003` отключены осознанно: кодовая база русскоязычная |
| Форматирование | `ruff format` | длина строки 100 |
| Докстроки | ruff `D` (google style) | обязательны для публичных модулей, классов, функций |
| Тесты | pytest | маркеры `scientific`, `benchmark`, `slow` |

Проверка на текущем коммите: `ruff` — чисто, `mypy --strict` — 0 ошибок в 28
файлах, `pytest` — 84 теста.

## 4. Как добавлять текст в интерфейс

**Жёстко зашитые строки запрещены** (§3 ТЗ, ADR-007).

1. Добавьте ключ в `src/quantumlab/i18n/locales/ru.json` **и** `en.json`;
2. Используйте `t("ключ", locale, параметр=значение)`;
3. Запустите `pytest tests/test_i18n_parity.py` — он проверит паритет ключей и
   совпадение плейсхолдеров.

Соглашения о ключах:

```text
nav.*            элементы навигации
wizard.*         шаги мастера
status.*         состояния задания
task.*           типы задач
profile.*        профили точности и их обоснования
tooltip.<p>.*    объяснимость (title/what/why/if_changed/relevant_for)
error.<код>.*    ошибки (title/what/hint)
action.*         кнопки в блоке «Что можно сделать»
attempt.*        стратегии SCF/оптимизации для блока «Что мы попробовали»
cli.*            вывод командной строки
```

## 5. Как добавлять ошибку

1. Код в `ErrorCode` (`src/quantumlab/errors.py`) — он попадает в API и аудит,
   переименовывать нельзя;
2. Класс ошибки: `QuantumLabError` для простой или `DiagnosisError`, если есть
   испробованные стратегии и действия;
3. Ключи `error.<код>.title`, `error.<код>.what`, опционально `.hint` — в оба
   каталога;
4. Тест: ошибка содержит ожидаемые параметры и локализуется.

Пример — `ScfNotConvergedError`: он несёт `attempts` (что реально применено) и
`actions` (кнопки), поэтому UI показывает правду, а не заготовленный текст.

## 6. Как добавлять метод, функционал или базис

1. Реализуйте Protocol из `quantumlab/engine/contracts.py`;
2. Зарегистрируйте `Capability` в реестре **со статусом `not_implemented`**;
3. Добавьте верификационные кейсы (§26 ТЗ);
4. Только после их прохождения поменяйте статус на `implemented` и укажите
   `since_version`.

Нарушение порядка (статус раньше верификации) — это прямое нарушение §54 ТЗ.

## 7. Как добавлять поле в спецификацию

1. Поле в `CalculationSpec` (или вложенной модели) с ограничением (`ge`, `gt`);
2. Обновите OpenAPI-схему `api/openapi/v1.yaml`;
3. Если поле влияет на результат — оно автоматически попадёт в отпечаток
   (`canonical_json`), миграция не нужна;
4. Проверьте `tests/test_openapi_contract.py` и `tests/test_fingerprint.py`.

Никогда не добавляйте «настройку по умолчанию, зашитую в движке»: всё, что
влияет на результат, должно быть в спецификации.

## 8. Как добавлять endpoint

1. Схема в OpenAPI → 2. handler в `server/api` (тонкий, без бизнес-логики) →
3. вызов доменного сервиса → 4. тест контракта и интеграционный тест.
Ошибки возвращаются как `Problem` с кодом из таксономии.

## 9. Коммиты и ревью

- Заголовок в повелительном наклонении, до 72 символов;
- В описании — **что и почему**, а не перечисление строк;
- Изменение поведения ядра сопровождается записью в ADR и обновлением
  верификационных кейсов;
- CI: lint → typecheck → unit → integration → scientific → perf → security →
  build. Красный научный тест блокирует слияние.

## 10. Отладка

```bash
# Детальный вывод тестов
pytest -q tests/test_recommend_profiles.py -x

# Проверить, что видит пользователь
quantumlab --lang ru plan tests/fixtures/water.xyz --task freq --profile research

# Состояние реестра
quantumlab capabilities --kind method

# Локальное хранилище заданий
ls ~/.quantumlab/jobs
```

## 11. Что читать дальше

- [`../architecture/06-quantum-engine.md`](../architecture/06-quantum-engine.md) —
  устройство ядра и требования к производительности;
- [`../architecture/09-verification-and-benchmark.md`](../architecture/09-verification-and-benchmark.md) —
  как писать верификационные кейсы и бенчмарки;
- [`../architecture/adr/`](../architecture/adr/) — почему приняты ключевые решения.
