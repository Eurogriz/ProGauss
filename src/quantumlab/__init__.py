"""QuantumLab — production-grade квантовохимическая платформа.

Пакет намеренно разбит на слои, которые связаны только через контракты
(``quantumlab.engine.contracts``) и доменные модели (``quantumlab.domain``):

* :mod:`quantumlab.domain`    — доменные модели (молекула, спецификация, задание, результат);
* :mod:`quantumlab.i18n`      — локализация (ru — язык по умолчанию, en — обязательный второй язык);
* :mod:`quantumlab.errors`    — человеко-понятная таксономия ошибок и диагностик;
* :mod:`quantumlab.engine`    — контракты расчётного ядра + реестр возможностей;
* :mod:`quantumlab.jobs`      — жизненный цикл задания (атомарные переходы состояний);
* :mod:`quantumlab.recommend` — автоподбор параметров («Рекомендуемые настройки»).

Ключевой принцип: GUI, CLI, REST API и Python SDK используют один и тот же
backend, поэтому все бизнес-правила живут здесь, а не в интерфейсных слоях.
"""

from quantumlab.version import __version__, api_version

__all__ = ["__version__", "api_version"]
