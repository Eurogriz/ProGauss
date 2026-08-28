"""Реестр возможностей — единственный источник правды о том, что реализовано.

GUI, CLI, REST API и Python SDK спрашивают у реестра, доступна ли возможность.
Это исключает ситуацию, когда интерфейс обещает метод, которого в ядре нет
(§54 ТЗ), и позволяет плагинам расширять систему без изменения интерфейсов.

.. warning::
   ``default_registry()`` описывает **текущее состояние репозитория**. Пока
   расчётное ядро не реализовано и не верифицировано, все вычислительные
   возможности имеют статус ``NOT_IMPLEMENTED``. Менять статус на
   ``IMPLEMENTED`` разрешено только вместе с прохождением верификационного
   набора (§26 ТЗ).
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence as Seq

from quantumlab.domain.spec import DispersionCorrection, Task
from quantumlab.engine.basis import basis_angular_scheme
from quantumlab.engine.capabilities import Availability, Capability, CapabilityKind
from quantumlab.engine.functional import FUNCTIONALS, get_functional
from quantumlab.errors import (
    BasisNotFoundError,
    FunctionalNotFoundError,
    MethodNotAvailableError,
)
from quantumlab.version import __version__


def _kind_from_identifier(identifier: str) -> CapabilityKind:
    """Определяет категорию по префиксу идентификатора (``basis:…``)."""
    prefix, _, _ = identifier.partition(":")
    try:
        return CapabilityKind(prefix)
    except ValueError:
        return CapabilityKind.METHOD


def _normalize(raw: str) -> str:
    return raw.strip().lower().replace(" ", "").replace("_", "-")


class CapabilityRegistry:
    """Потокобезопасный на чтение реестр возможностей.

    Реестр иммутабелен после сборки; регистрация плагинами выполняется на старте
    процесса до публикации реестра, поэтому блокировки не нужны.
    """

    def __init__(self, capabilities: Iterable[Capability] = ()) -> None:
        """Создаёт реестр из набора возможностей."""
        self._by_id: dict[str, Capability] = {}
        self._lookup: dict[str, str] = {}
        for capability in capabilities:
            self.register(capability)

    # -- регистрация --------------------------------------------------------- #
    def register(self, capability: Capability, *, replace: bool = False) -> None:
        """Добавляет возможность; дубликат без ``replace`` — ошибка."""
        if capability.id in self._by_id and not replace:
            msg = f"Возможность {capability.id!r} уже зарегистрирована"
            raise ValueError(msg)
        self._by_id[capability.id] = capability
        self._lookup[_normalize(capability.name)] = capability.id
        for alias in capability.aliases:
            self._lookup[_normalize(alias)] = capability.id

    # -- чтение -------------------------------------------------------------- #
    def get(self, identifier: str) -> Capability:
        """Возвращает возможность по точному идентификатору."""
        return self._by_id[identifier]

    def find(self, raw: str) -> Capability | None:
        """Ищет по имени или псевдониму без учёта регистра и пробелов."""
        target = self._lookup.get(_normalize(raw))
        return self._by_id[target] if target else None

    def list_capabilities(
        self,
        kind: CapabilityKind | None = None,
        *,
        available_only: bool = False,
    ) -> tuple[Capability, ...]:
        """Список возможностей, опционально отфильтрованный."""
        items: Seq[Capability] = tuple(self._by_id.values())
        if kind is not None:
            items = tuple(item for item in items if item.kind == kind)
        if available_only:
            items = tuple(item for item in items if item.is_usable)
        return tuple(sorted(items, key=lambda item: item.id))

    def availability(self, identifier: str) -> Availability:
        """Статус возможности; для неизвестной — ``NOT_IMPLEMENTED``."""
        capability = self._by_id.get(identifier)
        return capability.availability if capability else Availability.NOT_IMPLEMENTED

    def is_available(self, identifier: str) -> bool:
        """Доступна ли возможность для реального расчёта."""
        return self.availability(identifier).is_usable

    def assert_available(self, identifier: str) -> Capability:
        """Возвращает возможность или бросает понятную ошибку (§19 ТЗ).

        Тип ошибки зависит от категории: так GUI может показать «базис не
        найден» и «метод недоступен» по-разному и предложить разные действия.
        """
        capability = self._by_id.get(identifier)
        if capability is not None and capability.is_usable:
            return capability
        name = capability.name if capability else identifier
        kind = capability.kind if capability else _kind_from_identifier(identifier)
        if kind is CapabilityKind.BASIS:
            raise BasisNotFoundError(name)
        if kind is CapabilityKind.FUNCTIONAL:
            raise FunctionalNotFoundError(name)
        raise MethodNotAvailableError(name)

    def snapshot(self) -> dict[str, list[dict[str, object]]]:
        """Срез реестра для REST API и раздела «База методов» в GUI."""
        grouped: dict[str, list[dict[str, object]]] = {}
        for capability in self.list_capabilities():
            grouped.setdefault(capability.kind.value, []).append(
                {
                    "id": capability.id,
                    "name": capability.name,
                    "availability": capability.availability.value,
                    "since_version": capability.since_version,
                    "limitations": list(capability.limitations),
                    "metadata": dict(capability.metadata),
                }
            )
        return grouped

    def __len__(self) -> int:
        """Число зарегистрированных возможностей."""
        return len(self._by_id)


# --------------------------------------------------------------------------- #
# Состав по умолчанию: всё, что заявлено в ТЗ, с честным статусом реализации.
# --------------------------------------------------------------------------- #
_METHODS: tuple[tuple[str, str], ...] = (
    ("hf", "Hartree–Fock (RHF/UHF/ROHF)"),
    ("dft", "DFT (LDA/GGA/meta-GGA/hybrid/RSH/double-hybrid)"),
    ("mp2", "MP2"),
    ("scs_mp2", "SCS-MP2"),
    ("ccsd", "CCSD"),
    ("ccsd_t", "CCSD(T)"),
)

#: SVWN реализован и сверен с LibXC; остальные заявлены в ТЗ, но кода нет.
#: Статус записан рядом с именем, чтобы «заявлено» и «умеет» не разъезжались.
_FUNCTIONALS: tuple[tuple[str, str, str], ...] = (
    ("svwn", "SVWN (Слейтер + VWN-5)", "lda"),
    ("lda", "LDA (синоним SVWN)", "lda"),
    ("pbe", "PBE", "gga"),
    ("blyp", "BLYP", "gga"),
    ("pbe0", "PBE0", "hybrid"),
    ("b3lyp", "B3LYP", "hybrid"),
    ("tpssh", "TPSSh", "mgga"),
    ("m06", "M06", "mgga"),
    ("m062x", "M06-2X", "mgga"),
    ("wb97x", "ωB97X", "range_separated_hybrid"),
    ("wb97x-d", "ωB97X-D", "range_separated_hybrid"),
)

_BASIS_SETS: tuple[str, ...] = (
    "sto-3g",
    "3-21g",
    "6-31g",
    "6-31g(d)",
    "6-31g(d,p)",
    "6-311g",
    "6-311g(d,p)",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "aug-cc-pvdz",
    "aug-cc-pvtz",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
)

_FORMATS: tuple[tuple[str, Availability], ...] = (
    ("xyz", Availability.IMPLEMENTED),
    ("mol", Availability.NOT_IMPLEMENTED),
    ("sdf", Availability.NOT_IMPLEMENTED),
    ("pdb", Availability.NOT_IMPLEMENTED),
    ("cif", Availability.NOT_IMPLEMENTED),
    ("mol2", Availability.NOT_IMPLEMENTED),
    ("smiles", Availability.NOT_IMPLEMENTED),
    ("inchi", Availability.NOT_IMPLEMENTED),
)

#: Обработка спина и её ограничения. Статус partial означает «считает, но не
#: всё»: ограничения видны и в реестре, и в предупреждениях результата (§54 ТЗ).
_SPINS: tuple[tuple[str, Availability, tuple[str, ...]], ...] = (
    (
        "rhf",
        Availability.PARTIAL,
        ("Только замкнутая оболочка: нечётное число электронов требует UHF или ROHF.",),
    ),
    (
        "uhf",
        Availability.PARTIAL,
        (
            "Задачи: энергия в точке, оптимизация геометрии и частоты "
            "(переходные состояния и сканирования не реализованы).",
            "Оптимизация — только в декартовых координатах.",
            "Спиновое загрязнение возможно: значение <S^2> выводится в "
            "результат и в предупреждения, а не замалчивается.",
            "Однодетерминантное описание: при заметном спиновом загрязнении "
            "(<S^2> заметно выше S(S+1)) результат требует проверки.",
        ),
    ),
    ("rohf", Availability.NOT_IMPLEMENTED, ()),
)

#: Системы координат оптимизации. Реализованы только декартовы: избыточные
#: внутренние требуют матрицы Вильсона и её псевдообращения — отдельная задача.
_COORDINATES: tuple[tuple[str, Availability], ...] = (
    ("cartesian", Availability.PARTIAL),
    ("internal", Availability.NOT_IMPLEMENTED),
    ("redundant_internal", Availability.NOT_IMPLEMENTED),
)

_BACKENDS: tuple[tuple[str, Availability], ...] = (
    ("reference-cpu", Availability.IMPLEMENTED),
    ("optimized-cpu", Availability.NOT_IMPLEMENTED),
    ("cuda", Availability.NOT_IMPLEMENTED),
    ("rocm", Availability.NOT_IMPLEMENTED),
)

#: Ограничение, общее для всех базисов с d/f-функциями: движок считает в
#: декартовой схеме, тогда как эти наборы опубликованы в сферической.
_CARTESIAN_LIMITATION = (
    "Базис опубликован в сферической схеме (чистые угловые моменты), а расчёт "
    "идёт в декартовой: 6 d-функций вместо 5, 10 f вместо 7. Это больший базис, "
    "энергия ниже табличной примерно на 1e-4 Eh."
)

_SCHEDULERS: tuple[str, ...] = ("local", "slurm", "pbs", "lsf")


def _method_limitations(name: str) -> tuple[str, ...]:
    """Ограничения метода, видимые и в реестре, и в предупреждениях (§54 ТЗ)."""
    if name == "hf":
        return (
            "Только RHF: нечётное число электронов отклоняется.",
            "Задачи: энергия в точке, оптимизация геометрии и частоты "
            "(переходные состояния и сканирования не реализованы).",
            "Оптимизация — только в декартовых координатах.",
        )
    if name == "dft":
        return (
            "Реализованы SVWN (LDA), PBE (GGA) и PBE0 (гибрид, ¼ точного "
            "обмена); meta-GGA, B3LYP и дальнодействующие гибриды "
            "не реализованы.",
            "Дисперсионные поправки (D3, D4) не реализованы.",
            "Только RKS: UKS для открытой оболочки не реализован.",
            "Оптимизация геометрии реализована, но аналитический градиент не "
            "содержит отклика квадратурной сетки: расхождение с поверхностью "
            "оптимизатора измерено (7.0e-06 э/бор на воде/STO-3G) и в 64 раза "
            "ниже порога сходимости по силе.",
        )
    return ()


def _functional_limitations(name: str) -> tuple[str, ...]:
    """Ограничения конкретного функционала, а не общие для всех.

    Общая формулировка вроде «только LDA» для PBE и PBE0 была бы неправдой, а
    реестр — это то, на что пользователь опирается при выборе метода. Поэтому
    класс берётся из самого объекта функционала (§54 ТЗ).
    """
    functional = get_functional(name)
    limits: list[str] = []
    if functional.functional_class == "lda":
        limits.append("LDA: зависит только от плотности, без её градиента.")
    elif functional.functional_class == "gga":
        limits.append(
            "GGA: зависит от плотности и её градиента; кинетическая плотность "
            "(meta-GGA) не используется."
        )
    else:
        limits.append(
            f"Гибрид: {functional.exact_exchange_fraction:g} точного обмена; "
            "дальнодействующая коррекция (ωB97X и подобные) не реализована."
        )
    limits.append("Только замкнутая оболочка: UKS не реализован.")
    return tuple(limits)


#: Параметры SCF. Имена совпадают со значениями ``ScfSpec.fallback_strategies``,
#: иначе проверка «поддерживается ли запрошенное» сравнивала бы строки, которых
#: нет ни в спецификации, ни в реестре.
_SCF_OPTIONS: tuple[tuple[str, bool, str], ...] = (
    ("diis", True, ""),
    ("damping", True, ""),
    ("level_shift", True, ""),
    ("ediis", False, "EDIIS не реализован; запрос отклоняется, а не выполняется без него."),
    ("soscf", False, "SOSCF (второй порядок) не реализован."),
    ("stability_analysis", False, "Проверка устойчивости волновой функции не реализована."),
    ("fractional_occupations", False, "Дробные занятия не реализованы."),
)

#: Параметры оптимизатора. ``hessian_update`` объявлен в спецификации как
#: ``bfgs|bofill|none``, но движок всегда применяет BFGS — поэтому bofill и none
#: заявлены как нереализованные: иначе запрос «обновляй по Бофиллу» молча
#: выполнился бы с BFGS.
_OPTIMIZER_OPTIONS: tuple[tuple[str, bool, str], ...] = (
    ("frozen_atoms", True, ""),
    ("hessian_update:bfgs", True, ""),
    ("constraints", False, "Ограничения координат не реализованы."),
    (
        "hessian_update:bofill",
        False,
        "Обновление Бофилла не реализовано; движок всегда применяет BFGS.",
    ),
    ("hessian_update:none", False, "Отказ от обновления гессиана не реализован."),
)


def default_registry() -> CapabilityRegistry:
    """Собирает реестр, соответствующий текущему состоянию кодовой базы.

    Статусы отражают фактическое состояние кода:

    * ``implemented`` — реализовано и прошло верификацию (XYZ, single_point,
      reference-cpu, 6 декартовых базисов);
    * ``partial`` — работает с явными ограничениями (hf: только RHF и только
      single_point; 10 базисов со сферической публикацией d/f; spin:rhf);
    * ``not_implemented`` — заявлено в архитектуре, кода нет.

    Угловая схема базиса читается из самих данных, чтобы реестр не хранил
    второе, способное разойтись мнение о том же факте.
    """
    capabilities: list[Capability] = []

    for task in Task:
        # single_point проверен сверкой с PySCF (до 1e-6 Eh), optimization —
        # сверкой аналитического градиента с конечными разностями (до 1e-6 э/бор),
        # frequencies — сверкой гессиана с аналитическим гессианом PySCF
        # (до 1e-6 э/бор²) и совпадением самих частот.
        if task in (Task.SINGLE_POINT, Task.OPTIMIZATION):
            availability = Availability.IMPLEMENTED
        elif task is Task.FREQUENCIES:
            # partial, а не implemented: гессиан численный (центральные разности
            # аналитического градиента), и задача доступна только тем методам,
            # у которых аналитический градиент есть.
            availability = Availability.PARTIAL
        else:
            availability = Availability.NOT_IMPLEMENTED
        limitations: tuple[str, ...] = ()
        if task is Task.FREQUENCIES:
            limitations = (
                "Гессиан численный: центральные разности аналитического градиента, "
                "а не аналитические вторые производные.",
                "Доступны только методы с аналитическим градиентом: RHF, UHF, "
                "RKS со SVWN/PBE/PBE0.",
                "Стоимость — 6N расчётов градиента, поэтому задача заметно дороже одноточечной.",
            )
        capabilities.append(
            Capability(
                id=f"task:{task.value}",
                kind=CapabilityKind.TASK,
                name=task.value,
                availability=availability,
                since_version=__version__ if availability.is_usable else None,
                limitations=limitations,
            )
        )

    for name, label in _METHODS:
        # hf реализован, но только в варианте RHF и только для двух задач —
        # поэтому partial с явным перечнем ограничений, а не implemented.
        availability = (
            Availability.PARTIAL if name in ("hf", "dft") else Availability.NOT_IMPLEMENTED
        )
        limitations = _method_limitations(name)
        capabilities.append(
            Capability(
                id=f"method:{name}",
                kind=CapabilityKind.METHOD,
                name=name,
                availability=availability,
                since_version=__version__ if availability.is_usable else None,
                limitations=limitations,
                metadata={"label": label},
            )
        )

    for name, label, functional_class in _FUNCTIONALS:
        # Истина берётся из самого модуля функционалов: если реализация появится
        # или исчезнет, реестр изменится вместе с ней, а не по чьей-то памяти.
        implemented = name in FUNCTIONALS
        capabilities.append(
            Capability(
                id=f"functional:{name}",
                kind=CapabilityKind.FUNCTIONAL,
                name=name,
                availability=(
                    Availability.PARTIAL if implemented else Availability.NOT_IMPLEMENTED
                ),
                since_version=__version__ if implemented else None,
                limitations=(_functional_limitations(name) if implemented else ()),
                metadata={"label": label, "class": functional_class},
            )
        )

    for name in _BASIS_SETS:
        # Все 16 наборов загружаются и работают. Статус зависит от угловой
        # схемы, в которой набор опубликован: для сферических наш декартов
        # расчёт даёт больший базис, и это ограничение нужно показывать.
        cartesian = basis_angular_scheme(name) == "cartesian"
        capabilities.append(
            Capability(
                id=f"basis:{name}",
                kind=CapabilityKind.BASIS,
                name=name,
                availability=Availability.IMPLEMENTED if cartesian else Availability.PARTIAL,
                since_version=__version__,
                limitations=() if cartesian else (_CARTESIAN_LIMITATION,),
                metadata={"angular_scheme_published": "cartesian" if cartesian else "spherical"},
            )
        )

    for name, availability in _FORMATS:
        capabilities.append(
            Capability(
                id=f"format:{name}",
                kind=CapabilityKind.FORMAT,
                name=name,
                availability=availability,
                since_version=__version__ if availability.is_usable else None,
                notes_key=(
                    "capability.note.reference_engine"
                    if availability.is_usable
                    else "capability.note.not_implemented"
                ),
            )
        )

    for name, availability in _BACKENDS:
        capabilities.append(
            Capability(
                id=f"backend:{name}",
                kind=CapabilityKind.BACKEND,
                name=name,
                availability=availability,
                since_version=__version__ if availability.is_usable else None,
                limitations=(
                    ("Один поток, dense float64, O(N⁴) без скрининга.",)
                    if name == "reference-cpu"
                    else ()
                ),
            )
        )

    for name, availability in _COORDINATES:
        capabilities.append(
            Capability(
                id=f"coordinates:{name}",
                kind=CapabilityKind.COORDINATES,
                name=name,
                availability=availability,
                since_version=__version__ if availability.is_usable else None,
                limitations=(
                    (
                        "Сходимость медленнее, чем в избыточных внутренних "
                        "координатах: шесть нулевых мод (поступательные и "
                        "вращательные) ухудшают приближение гессиана.",
                    )
                    if name == "cartesian"
                    else ()
                ),
            )
        )

    for correction in DispersionCorrection:
        # Ни одна дисперсионная поправка не реализована: D3/D4 требуют
        # таблиц коэффициентов и своих параметров. Заявлять их доступными
        # означало бы выдать энергию без обещанной поправки (§54 ТЗ).
        capabilities.append(
            Capability(
                id=f"dispersion:{correction.value}",
                kind=CapabilityKind.DISPERSION,
                name=correction.value,
                availability=(
                    Availability.IMPLEMENTED
                    if correction is DispersionCorrection.NONE
                    else Availability.NOT_IMPLEMENTED
                ),
                since_version=__version__ if correction is DispersionCorrection.NONE else None,
            )
        )

    for spin_name, spin_availability, spin_limitations in _SPINS:
        capabilities.append(
            Capability(
                id=f"spin:{spin_name}",
                kind=CapabilityKind.SPIN,
                name=spin_name,
                availability=spin_availability,
                since_version=__version__ if spin_availability.is_usable else None,
                limitations=spin_limitations,
            )
        )

    for name in _SCHEDULERS:
        capabilities.append(
            Capability(
                id=f"scheduler:{name}",
                kind=CapabilityKind.SCHEDULER,
                name=name,
                availability=Availability.NOT_IMPLEMENTED,
            )
        )

    for name, available, note in _SCF_OPTIONS:
        capabilities.append(
            Capability(
                id=f"scf:{name}",
                kind=CapabilityKind.SCF,
                name=name,
                availability=(
                    Availability.IMPLEMENTED if available else Availability.NOT_IMPLEMENTED
                ),
                since_version=__version__ if available else None,
                limitations=() if available else (note,),
            )
        )

    for name, available, note in _OPTIMIZER_OPTIONS:
        capabilities.append(
            Capability(
                id=f"optimizer:{name}",
                kind=CapabilityKind.OPTIMIZER,
                name=name,
                availability=(
                    Availability.IMPLEMENTED if available else Availability.NOT_IMPLEMENTED
                ),
                since_version=__version__ if available else None,
                limitations=() if available else (note,),
            )
        )

    return CapabilityRegistry(capabilities)
