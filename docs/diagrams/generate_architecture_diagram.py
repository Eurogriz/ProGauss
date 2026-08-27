"""Генератор векторной диаграммы архитектуры.

Диаграмма хранится в репозитории как SVG, но генерируется скриптом — так она
не «зарастает» расхождениями с документацией: при изменении архитектуры
меняется этот файл, а картинка пересобирается.

Запуск::

    python docs/diagrams/generate_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

WIDTH = 1180
HEIGHT = 900

BACKGROUND = "#0f172a"
PANEL = "#1e293b"
PANEL_BORDER = "#334155"
ACCENT = "#38bdf8"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
WARN = "#f59e0b"

_PARTS: list[str] = []


def _rect(x: int, y: int, width: int, height: int, fill: str, stroke: str, radius: int = 10) -> None:
    _PARTS.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def _text(x: int, y: int, value: str, *, size: int = 14, fill: str = TEXT, anchor: str = "middle") -> None:
    _PARTS.append(
        f'<text x="{x}" y="{y}" font-family="Inter, Segoe UI, sans-serif" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def _box(x: int, y: int, width: int, height: int, title: str, subtitle: str = "") -> None:
    _rect(x, y, width, height, PANEL, PANEL_BORDER)
    _text(x + width // 2, y + height // 2 + (0 if not subtitle else -4), title, size=14)
    if subtitle:
        _text(x + width // 2, y + height // 2 + 16, subtitle, size=11, fill=MUTED)


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> None:
    _PARTS.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ACCENT}" '
        f'stroke-width="2" marker-end="url(#arrowhead)"/>'
    )
    if label:
        _text((x1 + x2) // 2 + 8, (y1 + y2) // 2, label, size=10, fill=MUTED, anchor="start")


def _section(x: int, y: int, width: int, height: int, title: str) -> None:
    _rect(x, y, width, height, "#111c33", PANEL_BORDER)
    _text(x + 14, y + 22, title, size=12, fill=ACCENT, anchor="start")


def build() -> str:
    """Собирает SVG-документ."""
    _PARTS.clear()
    _PARTS.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Inter, Segoe UI, sans-serif">'
    )
    _PARTS.append(
        f'<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        f'<path d="M0,0 L10,4 L0,8 z" fill="{ACCENT}"/></marker></defs>'
    )
    _PARTS.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>')
    _text(WIDTH // 2, 36, "QuantumLab — архитектура системы", size=22, fill=TEXT)
    _text(WIDTH // 2, 58, "UI → API → Job Manager → Scheduler → Worker → Quantum Engine → Result Store", size=12, fill=MUTED)

    # 1. Интерфейсы пользователя
    _section(40, 80, 780, 90, "ИНТЕРФЕЙСЫ ПОЛЬЗОВАТЕЛЯ")
    for index, (title, subtitle) in enumerate(
        [("Web UI", "React · TS · WebGL"), ("CLI", "quantumlab"), ("Python SDK", "batch · workflows"), ("Внешние системы", "ноутбуки, пайплайны")]
    ):
        _box(60 + index * 190, 110, 170, 48, title, subtitle)

    # 2. API
    _section(40, 195, 780, 90, "API-SERVER · FASTAPI · /api/v1")
    for index, title in enumerate(["AuthN/AuthZ · RBAC", "Projects", "Molecules", "Jobs", "Results", "Workers"]):
        _box(60 + index * 126, 225, 116, 44, title)

    # 3. Платформенное ядро
    _section(40, 310, 780, 100, "ЯДРО ПЛАТФОРМЫ (общий код для всех интерфейсов)")
    for index, (title, subtitle) in enumerate(
        [
            ("domain", "Molecule · Spec · Job"),
            ("recommend", "автоподбор + обоснования"),
            ("registry", "что реально реализовано"),
            ("i18n", "ru (по умолч.) · en"),
            ("job-manager", "состояния · retry · аудит"),
        ]
    ):
        _box(60 + index * 152, 342, 142, 52, title, subtitle)

    # 4. Планировщик
    _section(40, 435, 780, 90, "SCHEDULER")
    for index, title in enumerate(["Очередь · приоритеты", "local", "Slurm / PBS / LSF", "Kubernetes"]):
        _box(60 + index * 190, 465, 170, 44, title)

    # 5. Worker и ядро
    _section(40, 550, 780, 220, "WORKER-RUNTIME · QUANTUM ENGINE")
    _box(60, 580, 170, 60, "worker", "sandbox · лимиты · чекпоинты")
    _box(250, 580, 170, 60, "QuantumEngine", "фасад ядра")
    _box(440, 580, 170, 60, "molecule-engine", "basis-library")
    _box(630, 580, 170, 60, "checkpoint", "артефакты · логи")
    for index, title in enumerate(["integral", "scf", "dft", "correlation", "optimization", "frequency", "property"]):
        _box(60 + index * 106, 660, 96, 40, title)
    _box(60, 716, 740, 40, "backends: reference-cpu · optimized-cpu (OpenMP/BLAS) · CUDA · ROCm")

    # 6. Хранилище и наблюдаемость
    _section(840, 80, 300, 300, "ХРАНИЛИЩЕ")
    _box(860, 112, 260, 52, "PostgreSQL", "метаданные · статусы · аудит")
    _box(860, 176, 260, 52, "Object storage (S3)", "градиенты · орбитали · кубы")
    _box(860, 240, 260, 52, "Redis", "кэш · прогресс · блокировки")
    _box(860, 304, 260, 52, "Result Store", "версионируемые артефакты")

    _section(840, 400, 300, 190, "НАБЛЮДАЕМОСТЬ")
    _box(860, 432, 260, 44, "structured logs", "JSON · job_id")
    _box(860, 486, 260, 44, "metrics", "Prometheus")
    _box(860, 540, 260, 44, "tracing · health", "OpenTelemetry")

    _section(840, 610, 300, 160, "СТАТУС РЕАЛИЗАЦИИ")
    _rect(860, 640, 260, 52, "#1a2233", WARN)
    _text(990, 662, "Расчётное ядро не реализовано", size=12, fill=WARN)
    _text(990, 680, "реестр честно отдаёт not_implemented", size=10, fill=MUTED)
    _text(990, 720, "контрактный слой реализован и покрыт тестами", size=11, fill=MUTED)

    # Стрелки основного потока
    _arrow(430, 170, 430, 195)
    _arrow(430, 285, 430, 310)
    _arrow(430, 410, 430, 435)
    _arrow(430, 525, 430, 550)
    _arrow(430, 640, 430, 660)
    _arrow(430, 700, 430, 716)
    _arrow(820, 360, 860, 360, "метаданные")
    _arrow(820, 700, 860, 700)

    _PARTS.append("</svg>")
    return "\n".join(_PARTS)


def main() -> None:
    """Записывает SVG рядом со скриптом."""
    target = Path(__file__).with_name("architecture-overview.svg")
    target.write_text(build(), encoding="utf-8")
    print(f"записано: {target}")


if __name__ == "__main__":
    main()
