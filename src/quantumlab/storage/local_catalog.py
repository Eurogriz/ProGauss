"""Локальный каталог проектов и структур.

REST-контракт принимает ``moleculeId``, а не текст структуры, поэтому структуры
нужно где-то хранить до расчёта. Это файловая реализация того же уровня, что
:class:`~quantumlab.storage.local_jobs.LocalJobStore`: один JSON на сущность,
атомарная замена файла, никакого внешнего сервера.

Разделение по §24 ТЗ здесь условное: метаданные и структура лежат рядом,
потому что структура в XYZ занимает килобайты. Переезд на SQL + объектное
хранилище меняет реализацию, но не интерфейс.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from quantumlab.domain.molecule import Molecule
from quantumlab.errors import CatalogEntryNotFoundError, UnsupportedStructureFormatError


class ProjectRecord(BaseModel):
    """Проект — граница изоляции структур и заданий (§25 ТЗ)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=200)
    role: str = "owner"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MoleculeRecord(BaseModel):
    """Сохранённая структура вместе с её идентификатором и проектом."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    name: str
    format: str
    charge: int = 0
    multiplicity: int = 1
    molecule: Molecule
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


#: Форматы, которые каталог действительно умеет разбирать. Остальные заявлены
#: в архитектуре и честно отклоняются — молча принимать их нельзя (§54 ТЗ).
PARSABLE_FORMATS: frozenset[str] = frozenset({"xyz"})


class LocalCatalog:
    """Файловое хранилище проектов и структур."""

    def __init__(self, root: Path) -> None:
        """Создаёт каталог в ``root``, при необходимости создавая каталоги."""
        self.root = root
        self.projects_dir = root / "projects"
        self.molecules_dir = root / "molecules"
        for directory in (self.projects_dir, self.molecules_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # -- проекты ---------------------------------------------------------- #
    def create_project(self, name: str) -> ProjectRecord:
        """Создаёт проект и возвращает его запись."""
        record = ProjectRecord(id=str(uuid.uuid4()), name=name)
        self._write(self.projects_dir / f"{record.id}.json", record.model_dump_json(indent=2))
        return record

    def get_project(self, project_id: str) -> ProjectRecord:
        """Возвращает проект или бросает :class:`CatalogEntryNotFoundError`."""
        path = self.projects_dir / f"{project_id}.json"
        if not path.exists():
            raise CatalogEntryNotFoundError("project", project_id)
        return ProjectRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> tuple[ProjectRecord, ...]:
        """Все проекты, упорядоченные по имени."""
        return tuple(
            sorted(
                (
                    ProjectRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    for path in self.projects_dir.glob("*.json")
                ),
                key=lambda item: item.name,
            )
        )

    # -- структуры -------------------------------------------------------- #
    def create_molecule(
        self,
        *,
        project_id: str,
        name: str | None,
        content: str,
        fmt: str = "xyz",
        charge: int = 0,
        multiplicity: int = 1,
    ) -> MoleculeRecord:
        """Разбирает структуру и сохраняет её в каталоге.

        Проект обязан существовать: структура без проекта нарушила бы
        изоляцию (§25 ТЗ).
        """
        self.get_project(project_id)
        if fmt not in PARSABLE_FORMATS:
            raise UnsupportedStructureFormatError(name=fmt)
        # Заряд и мультиплетность обязаны попасть в саму структуру: число
        # электронов считается от неё, и если оставить их только в записи,
        # расчёт пойдёт по нейтральной молекуле молча.
        molecule = Molecule.from_xyz(content, name=name or "molecule").with_state(
            charge=charge, multiplicity=multiplicity
        )
        record = MoleculeRecord(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=name or molecule.formula,
            format=fmt,
            charge=charge,
            multiplicity=multiplicity,
            molecule=molecule,
        )
        self._write(self.molecules_dir / f"{record.id}.json", record.model_dump_json(indent=2))
        return record

    def get_molecule(self, molecule_id: str) -> MoleculeRecord:
        """Возвращает структуру или бросает :class:`CatalogEntryNotFoundError`."""
        path = self.molecules_dir / f"{molecule_id}.json"
        if not path.exists():
            raise CatalogEntryNotFoundError("molecule", molecule_id)
        return MoleculeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_molecules(self, project_id: str) -> tuple[MoleculeRecord, ...]:
        """Структуры проекта, упорядоченные по имени."""
        return tuple(
            sorted(
                (
                    record
                    for record in (
                        MoleculeRecord.model_validate_json(path.read_text(encoding="utf-8"))
                        for path in self.molecules_dir.glob("*.json")
                    )
                    if record.project_id == project_id
                ),
                key=lambda item: item.name,
            )
        )

    # -- служебное -------------------------------------------------------- #
    @staticmethod
    def _write(target: Path, payload: str) -> None:
        """Пишет атомарно: читатель не должен увидеть половинy файла."""
        temporary = target.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(target)

    def __len__(self) -> int:
        """Число сохранённых структур."""
        return len(list(self.molecules_dir.glob("*.json")))
