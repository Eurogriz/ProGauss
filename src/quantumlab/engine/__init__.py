"""Расчётное ядро: контракты, реестр возможностей, автоподбор параметров.

Модуль ``contracts`` описывает **интерфейсы**, а не реализации: любое ядро
(reференсное NumPy-ядро, будущий C++/CUDA-движок, внешний пакет через адаптер)
подключается, реализовав :class:`~quantumlab.engine.contracts.QuantumEngine`.

``registry`` — единственный источник правды о том, что реально реализовано
(§54 ТЗ): UI, CLI и API спрашивают у реестра, а не «знают» список методов.
"""

from quantumlab.engine.capabilities import Availability, Capability, CapabilityKind
from quantumlab.engine.contracts import (
    Array,
    BackendCapabilities,
    CheckpointHandle,
    CheckpointStore,
    ComputeBackend,
    CorrelationEngine,
    EngineRequest,
    ExchangeCorrelationFunctional,
    FrequencyEngine,
    GradientEngine,
    IntegralEngine,
    OptimizerEngine,
    ProgressReporter,
    PropertyEngine,
    QuantumEngine,
    ScfResult,
    ScfSolver,
)
from quantumlab.engine.registry import CapabilityRegistry, default_registry

__all__ = [
    "Array",
    "Availability",
    "BackendCapabilities",
    "Capability",
    "CapabilityKind",
    "CapabilityRegistry",
    "CheckpointHandle",
    "CheckpointStore",
    "ComputeBackend",
    "CorrelationEngine",
    "EngineRequest",
    "ExchangeCorrelationFunctional",
    "FrequencyEngine",
    "GradientEngine",
    "IntegralEngine",
    "OptimizerEngine",
    "ProgressReporter",
    "PropertyEngine",
    "QuantumEngine",
    "ScfResult",
    "ScfSolver",
    "default_registry",
]
