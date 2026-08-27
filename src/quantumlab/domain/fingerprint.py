"""Отпечаток расчёта (calculation fingerprint) — основа воспроизводимости (§40 ТЗ).

Отпечаток — это SHA-256 от канонического представления **всего**, что влияет
на результат: версии кода и движка, спецификации, исходной структуры, железа,
программного окружения и (для завершённого расчёта) финальной геометрии.

Два разных расчёта с одинаковым отпечатком обязаны дать одинаковый результат
в пределах объявленного порога сходимости. На этом инварианте держатся:
регрессионные тесты, кэш результатов, проверка «тот ли это расчёт» в отчёте
и аудит.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from quantumlab.domain.molecule import Molecule
from quantumlab.domain.spec import CalculationSpec


class Fingerprint(BaseModel):
    """Результат вычисления отпечатка."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    digest: str = Field(description="Полный SHA-256 в hex")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Дайджесты отдельных компонентов — для диагностики расхождений",
    )

    @property
    def short(self) -> str:
        """Короткая форма для UI и имени файла отчёта."""
        return self.digest[:12]


def _sha256(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_fingerprint(
    *,
    spec: CalculationSpec,
    molecule: Molecule,
    software_version: str,
    engine_version: str,
    hardware: Mapping[str, str] | None = None,
    environment: Mapping[str, str] | None = None,
    final_molecule: Molecule | None = None,
) -> Fingerprint:
    """Собирает отпечаток расчёта из всех влияющих на результат компонентов.

    Args:
        spec: спецификация расчёта.
        molecule: исходная структура.
        software_version: версия QuantumLab.
        engine_version: версия расчётного ядра (может отличаться от версии платформы).
        hardware: описание железа (модель CPU, число ядер, GPU, …).
        environment: программное окружение (ОС, BLAS, компилятор, MPI).
        final_molecule: итоговая геометрия, если расчёт её менял (оптимизация).
    """
    components = {
        "software_version": _sha256(software_version),
        "engine_version": _sha256(engine_version),
        "spec": _sha256(spec.canonical_json()),
        "initial_structure": molecule.structure_hash(),
        "hardware": _sha256(dict(hardware or {})),
        "environment": _sha256(dict(environment or {})),
        "seed": _sha256(spec.seed),
        "final_structure": final_molecule.structure_hash() if final_molecule else "",
    }
    return Fingerprint(digest=_sha256(components), components=components)
