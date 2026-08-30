# 05. Контракты: REST API, CLI, Python SDK

Все три интерфейса — оболочки над одним backend'ом (§20, §21, §22 ТЗ).
Контракт REST описан машиночитаемо в [`api/openapi/v1.yaml`](../../api/openapi/v1.yaml)
(OpenAPI 3.1, **23 пути, 26 операций, 30 схем**; валидность и согласованность
enum'ов с кодом проверяет `tests/test_openapi_contract.py`).

## 5.1 REST API `/api/v1`

| Группа | Операции |
|---|---|
| `auth` | `POST /auth/token`, `GET /auth/me` |
| `projects` | `GET/POST /projects`, участники и роли |
| `molecules` | `GET/POST /projects/{id}/molecules`, `GET /molecules/{id}`, `POST /molecules/{id}/validate` |
| `calculations` | `POST /calculations/plan`, `GET/POST /jobs`, `GET /jobs/{id}`, `cancel`, `resume`, `retry`, `events` (SSE), `logs` |
| `results` | `GET /jobs/{id}/result`, `GET /jobs/{id}/report`, `POST /jobs/{id}/result/export`, `GET /artifacts/{id}` |
| `workers` | `GET /workers` |
| `meta` | `GET /capabilities`, `GET /i18n/{locale}`, `GET /health`, `GET /ready` |

### Ключевые решения

**`POST /calculations/plan` — отдельный эндпоинт, а не часть `submit`.**
Это реализация требования §8 ТЗ «все автоматические решения должны
отображаться пользователю»: UI вызывает `plan`, показывает список
обоснований, и только потом отправляет `submit` с подтверждённой
спецификацией. Расчёт не может стартовать с параметрами, которые пользователь
не видел.

**Ошибки — RFC 9457 (`application/problem+json`) с расширением.**
Помимо стандартных полей возвращаются `code`, `hint`, `tried`, `actions` —
ровно то, что нужно для блоков «Что произошло / Что мы попробовали / Что можно
сделать» (§19 ТЗ). Кнопки действия приходят с сервера вместе с их
`kind` (`automatic | manual | navigate`), поэтому поведение кнопок одинаково
в Web UI, CLI и у внешних интеграций.

```json
{
  "type": "https://quantumlab.example/problems/scf-not-converged",
  "title": "Электронная структура не достигла требуемой сходимости",
  "status": 422,
  "code": "scf.not_converged",
  "detail": "Система пыталась найти устойчивое электронное состояние, но итерационный процесс не сошёлся за 80 итераций (остаточная ошибка 3.100e-05).",
  "hint": "Такое поведение характерно для систем с близкими по энергии орбиталями…",
  "tried": ["DIIS-ускорение", "затухание (damping)", "сдвиг уровней (level shifting)"],
  "actions": [
    { "key": "action.retry_automatic", "kind": "automatic", "label": "Повторить автоматически",
      "payload": { "profile": "robust_scf" } },
    { "key": "action.edit_settings", "kind": "manual", "label": "Изменить настройки вручную", "payload": {} }
  ]
}
```

**Тяжёлые данные не ходят телом ответа.** `GET /artifacts/{id}` отдаёт `302`
на подписанный URL объектного хранилища. Иначе API-сервер становится узким
местом на загрузке гессиана.

**Язык** — заголовок `Accept-Language` (`ru` по умолчанию), который влияет на
`title`, `detail`, `hint`, `tried`, `actions[].label` и `statusLabel`. Каталог
строк для UI отдаётся целиком через `GET /i18n/{locale}` — один источник для
всех интерфейсов.

**Версионирование.** `/api/v1` стабилен в пределах мажорной версии.
Несовместимое изменение → `/api/v2` и сосуществование минимум одного релизного
цикла (§47 ТЗ).

## 5.2 CLI (§20 ТЗ)

Реализован в `src/quantumlab/cli.py`, точка входа — консольный скрипт
`quantumlab`.

```bash
quantumlab --lang ru|en --data-dir ~/.quantumlab <команда>

quantumlab version
quantumlab capabilities [--kind method|functional|basis|task|format|backend|scheduler]
quantumlab molecule inspect <file.xyz> [--charge 0] [--multiplicity 1]
quantumlab plan  <file.xyz> --task optimize --profile high-accuracy
quantumlab run   <file.xyz> --task optimize --method dft --functional pbe0 --basis def2-tzvp
quantumlab job list [--status queued]
quantumlab job status|logs|cancel|resume|retry <id>
```

Принципы:

1. **Тот же backend.** CLI вызывает `resolve_profile`, `CapabilityRegistry` и
   `Job.transition_to` — те же функции, что и API. Никаких «CLI-исключений».
2. **Локализованный вывод** через тот же каталог: `--lang en` переключает
   сообщения, коды ошибок при этом не меняются.
3. **Код возврата отражает суть**: `0` — успех, `1` — задание создано, но
   выполнить нельзя (например, ядро не подключено), `2` — ошибка ввода/состояния.
4. **Машинная читаемость.** Для автоматизации добавляется `--json`
   (Этап 3): тот же вывод, но структурно.

Реальный вывод на текущем коммите:

```console
$ quantumlab plan water.xyz --task optimize --profile high-accuracy
Конфигурация расчёта
  Оптимизация геометрии
  Потому что выбран профиль «Высокая точность»
  Выбран функционал pbe0
  Выбран базисный набор def2-tzvp
  Дисперсионная поправка: d3bj
  Квадратурная сетка: ultrafine
  Порог сходимости SCF: 1e-09
  Порог отсечки интегралов: 1e-11
  Число потоков: 1
  Память на расчёт: 1024 МБ — оценка по 30 базисным функциям
  Вычислительное устройство: auto — доступно GPU: 0
```

## 5.3 Python SDK (§21 ТЗ)

Целевой интерфейс (соответствует требованию ТЗ и реализуется в Этапе 3):

```python
from quantumlab import Calculation, Molecule

mol = Molecule.from_smiles("CCO")  # формат доступен, когда реализован парсер

calc = Calculation(
    molecule=mol,
    task="optimization",
    method="DFT",
    functional="PBE0",
    basis="def2-TZVP",
)
result = calc.run()  # блокирующий вызов с прогрессом в лог

print(result.energy_hartree)
print(result.optimized_geometry.to_xyz())
print(result.quality_checks)  # проверки качества (§28 ТЗ)
```

Дополнительно для автоматизации:

```python
from quantumlab import Batch, plan

# Автоподбор с объяснениями — тот же код, что за кнопкой «Рассчитать»
resolution = plan(molecule=mol, task="frequencies", profile="high_accuracy")
for line in resolution.explain("ru"):
    print(line)

# Пакетный режим (§38 ТЗ): concurrency, retries, агрегированный экспорт
batch = Batch(spec=resolution.spec, retries=2, concurrency=16)
table = batch.run(molecules).to_dataframe()
table.to_parquet("screening.parquet")
```

Правила SDK:

- `Molecule`/`CalculationSpec` — те же pydantic-модели, что в ядре: объект из
  SDK можно сериализовать в JSON и отправить в API без преобразований;
- ошибки SDK — те же классы из `quantumlab.errors`, поэтому обработка в
  скрипте и в UI опирается на одни коды;
- SDK не выполняет тяжёлые вычисления в процессе пользователя по умолчанию:
  `calc.run()` отправляет задание и ждёт результата, локальный запуск —
  явная опция `backend="local"`.

## 5.4 Совместимость слоёв

| Изменение | Требуемое действие |
|---|---|
| Новое необязательное поле в `CalculationSpec` | Минорная версия; старая спецификация остаётся валидной |
| Новый код ошибки | Минорная; старые клиенты показывают `detail` |
| Изменение смысла поля / удаление поля | `/api/v2` + мигратор артефактов + запись в ADR |
| Новый метод/базис/функционал | Регистрация в реестре + верификация; API не меняется |
