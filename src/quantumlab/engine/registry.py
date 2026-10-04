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
from quantumlab.engine.basis_custom import custom_basis_names
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

#: Статус «реализован/заявлено» реестр читает из ``FUNCTIONALS`` (см. ниже):
#: здесь только справочные имена и классы, чтобы «заявлено» и «умеет» не
#: разъезжались. Реализованы SVWN, PBE, BLYP, PBE0, B3LYP (сверены с LibXC и
#: PySCF, в том числе со спиновой поляризацией) и meta-GGA гибрид TPSSh (только
#: RKS); M06, M06-2X и дальнодействующие гибриды — заявлены в
#: ТЗ, кода нет.
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
            "Для HF — это UHF; для DFT — спиново-поляризованный UKS "
            "(открытая оболочка считается с двумя канальными плотностями и "
            "аналитическим градиентом).",
        ),
    ),
    (
        "rohf",
        Availability.PARTIAL,
        (
            "Задачи: энергия в точке, оптимизация геометрии и частоты "
            "(переходные состояния и сканирования не реализованы).",
            "Оптимизация — только в декартовых координатах.",
            "Только для метода HF: для DFT ограниченная открытая оболочка не "
            "определена — открытооболочечный DFT считается как UKS "
            "(spin:uhf), и сочетание DFT + rohf отклоняется.",
            "Орбитальные энергии — собственные значения эффективного фокиана "
            "Рутаана; отдельных энергий каналов α и β он не даёт.",
            "В отличие от UHF состояние остаётся собственным для Ŝ², поэтому "
            "<S^2> = S(S+1) точно и спинового загрязнения нет.",
        ),
    ),
)

#: Системы координат оптимизации. Декартовы и избыточные внутренние
#: (``redundant_internal``: примитивы по связности, матрица Вильсона,
#: проекция ограничений); неизбыточные ``internal`` (Z-матрица) не реализованы.
_COORDINATES: tuple[tuple[str, Availability], ...] = (
    ("cartesian", Availability.PARTIAL),
    ("internal", Availability.NOT_IMPLEMENTED),
    ("redundant_internal", Availability.PARTIAL),
)

_COORDINATE_LIMITATIONS: dict[str, tuple[str, ...]] = {
    "cartesian": (
        "Сходимость медленнее, чем в избыточных внутренних координатах: шесть "
        "нулевых мод (поступательные и вращательные) ухудшают приближение гессиана.",
        "Ограничения координат (constraints) в декартовых координатах не "
        "поддерживаются; замороженные атомы — поддерживаются.",
    ),
    "redundant_internal": (
        "Набор примитивов (связи, углы, двугранные и линейные изгибы) строится "
        "один раз по исходной геометрии и не пересматривается: разрыв связи или "
        "прохождение угла через 180° в ходе оптимизации им не отслеживается.",
        "Ограничения: длина связи (Å), валентный угол в интервале 0–175° и "
        "двугранный угол (градусы); замороженные атомы вводятся декартовыми "
        "ограничениями. Начальный гессиан — модель Линдха, обновление — BFGS.",
    ),
}

_BACKENDS: tuple[tuple[str, Availability], ...] = (
    ("reference-cpu", Availability.IMPLEMENTED),
    ("optimized-cpu", Availability.NOT_IMPLEMENTED),
    ("cuda", Availability.NOT_IMPLEMENTED),
    ("rocm", Availability.NOT_IMPLEMENTED),
)

_SCHEDULERS: tuple[str, ...] = ("local", "slurm", "pbs", "lsf")


def _method_limitations(name: str) -> tuple[str, ...]:
    """Ограничения метода, видимые и в реестре, и в предупреждениях (§54 ТЗ)."""
    if name == "hf":
        return (
            "Замкнутая оболочка — RHF, открытая — UHF (возможное спиновое "
            "загрязнение, <S^2> выводится) или ROHF (точное <S^2>).",
            "Задачи: энергия в точке, оптимизация геометрии и частоты "
            "(переходные состояния и сканирования не реализованы).",
            "Оптимизация — только в декартовых координатах.",
        )
    if name == "dft":
        return (
            "Реализованы SVWN (LDA), PBE и BLYP (GGA), PBE0 и B3LYP (гибриды), "
            "TPSSh (meta-GGA гибрид, только замкнутая оболочка RKS); meta-GGA "
            "M06 и M06-2X и дальнодействующие гибриды (ωB97X, ωB97X-D) не "
            "реализованы.",
            "Дисперсионные поправки: D3 (BJ, zero) и D4 реализованы для "
            "функционалов с обученными параметрами.",
            "Замкнутая оболочка — RKS, открытая — спиново-поляризованный UKS "
            "(spin:uhf); для DFT нет ограниченной открытой оболочки "
            "(spin:rohf отклоняется).",
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
    elif functional.functional_class == "mgga":
        limits.append(
            "meta-GGA: зависит от плотности, её градиента и кинетической плотности τ; "
            f"доля точного обмена — {functional.exact_exchange_fraction:g}."
        )
        limits.append(
            "Только замкнутая оболочка (RKS): спин-поляризованный UKS с τ не "
            "реализован и отклоняется, а не подменяется приближением. Энергия, "
            "аналитический градиент, оптимизация и частоты — доступны."
        )
        return tuple(limits)
    else:
        limits.append(
            f"Гибрид: {functional.exact_exchange_fraction:g} точного обмена; "
            "дальнодействующая коррекция (ωB97X и подобные) не реализована."
        )
    limits.append(
        "Замкнутая оболочка — RKS, открытая — спиново-поляризованный UKS "
        "(spin:uhf); для DFT нет ограниченной открытой оболочки (spin:rohf)."
    )
    return tuple(limits)


#: Параметры SCF. Имена совпадают со значениями ``ScfSpec.fallback_strategies``,
#: иначе проверка «поддерживается ли запрошенное» сравнивала бы строки, которых
#: нет ни в спецификации, ни в реестре.
_SCF_OPTIONS: tuple[tuple[str, bool, str], ...] = (
    ("diis", True, ""),
    ("damping", True, ""),
    ("level_shift", True, ""),
    (
        "ediis",
        True,
        "EDIIS (Кудин–Скузерия–Кансес) для RHF, UHF, RKS и UKS; для ROHF запрос отклоняется.",
    ),
    ("soscf", False, "SOSCF (второй порядок) не реализован."),
    (
        "direct",
        True,
        "Прямой SCF: J и K собираются на лету без хранения тензора ERI, со скринингом "
        "Шварца 1e-12; инкрементальный Fock (по ΔD) не реализован, интегралы "
        "пересчитываются целиком на каждой итерации.",
    ),
    (
        "stability_analysis",
        True,
        "Одноточечный расчёт: RHF/RKS (внутренняя, RHF→UHF, мнимые вращения), "
        "UHF/UKS (внутренняя, мнимые вращения), ROHF (вещественные и мнимые "
        "вращения). DFT и ROHF — конечная разность орбитального градиента, HF — "
        "аналитические A±B. Не реализованы: meta-GGA, спин-неограниченные "
        "(GHF) вращения, анализ в оптимизации и частотах; в прямом SCF мнимые "
        "каналы не считаются.",
    ),
    ("fractional_occupations", False, "Дробные занятия не реализованы."),
)

#: Параметры SCF, реализованные частично (ограничения — в примечании).
_PARTIAL_SCF_OPTIONS: frozenset[str] = frozenset({"stability_analysis", "ediis", "direct"})

#: Управление заданиями. Доступность задаётся перечислением, а не булевым
#: флагом: контрольные точки реализованы не для всех расчётов (нет для частот),
#: и булев флаг не отличил бы это от «реализовано полностью». Повтор и
#: продолжение разделены: повтор выполняется заново, продолжение — с точки.
_JOB_OPTIONS: tuple[tuple[str, Availability, tuple[str, ...]], ...] = (
    ("retry", Availability.IMPLEMENTED, ()),
    (
        "checkpoint",
        Availability.PARTIAL,
        (
            "Пишутся для SCF в одной точке (RHF, UHF, ROHF, RKS, UKS) и для "
            "оптимизации геометрии; для частот контрольные точки не создаются, "
            "и «Продолжить расчёт» для них честно отказывает.",
            "SCF: восстанавливается плотность (для ROHF — через эффективный фокиан "
            "Рутаана), а не история DIIS; оптимизация: геометрия, приближение "
            "гессиана, шаг и журнал — продолжение даёт ту же траекторию.",
            "Рестарт SCF — продолжение с сохранённой волновой функции, а не "
            "побитовое воспроизведение прерванной последовательности итераций.",
            "Контрольная точка привязана к геометрии и базису: при их "
            "изменении рестарт отклоняется, а не выполняется с чужой плотностью.",
        ),
    ),
)

#: Параметры оптимизатора. ``hessian_update`` объявлен в спецификации как
#: ``bfgs|bofill|none``, но движок всегда применяет BFGS — поэтому bofill и none
#: заявлены как нереализованные: иначе запрос «обновляй по Бофиллу» молча
#: выполнился бы с BFGS.
_OPTIMIZER_OPTIONS: tuple[tuple[str, bool, str], ...] = (
    ("frozen_atoms", True, ""),
    ("hessian_update:bfgs", True, ""),
    (
        "constraints",
        True,
        "Только в системе координат redundant_internal: длина связи (Å), "
        "валентный и двугранный углы (градусы), либо удержание исходного значения.",
    ),
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
                "ROHF, RKS и UKS (SVWN, PBE, BLYP, PBE0, B3LYP).",
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
        # hf и dft реализованы, но не во всех вариантах: hf — RHF/UHF/ROHF,
        # dft — RKS/UKS с ограниченным набором функционалов, и оба — только для
        # двух задач. Поэтому partial с явным перечнем ограничений, а не
        # implemented.
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
                notes_key=(
                    "capability.note.partial"
                    if availability is Availability.PARTIAL
                    else "capability.note.not_implemented"
                ),
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
                notes_key=(
                    "capability.note.partial" if implemented else "capability.note.not_implemented"
                ),
            )
        )

    for name in _BASIS_SETS:
        # Все 16 наборов загружаются и считаются в той угловой схеме, в которой
        # опубликованы (сферической или декартовой), поэтому энергии
        # воспроизводят табличные.
        scheme = basis_angular_scheme(name)
        capabilities.append(
            Capability(
                id=f"basis:{name}",
                kind=CapabilityKind.BASIS,
                name=name,
                availability=Availability.IMPLEMENTED,
                since_version=__version__,
                metadata={"angular_scheme_published": scheme, "angular_scheme_used": scheme},
            )
        )

    # Пользовательские базисы (каталог пользователя, см. basis_custom): те же
    # данные и тот же код, поэтому ``implemented``; происхождение — в метаданных.
    for name in custom_basis_names():
        if name in _BASIS_SETS:
            continue
        try:
            scheme = basis_angular_scheme(name)
        except (BasisNotFoundError, ValueError, KeyError):
            continue
        capabilities.append(
            Capability(
                id=f"basis:{name}",
                kind=CapabilityKind.BASIS,
                name=name,
                availability=Availability.IMPLEMENTED,
                since_version=__version__,
                metadata={
                    "angular_scheme_published": scheme,
                    "angular_scheme_used": scheme,
                    "origin": "custom",
                },
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
                limitations=_COORDINATE_LIMITATIONS.get(name, ()),
            )
        )

    for correction in DispersionCorrection:
        # D3 и D4 реализованы, но область применения уже, чем у остальных
        # методов: у D3 — элементы H–F, Si, P, S, Cl, Br, I и функционалы
        # (hf, pbe, pbe0, blyp, b3lyp); у D4 — обученные функционалы dftd4
        # v4.0.1 (включая hf). За это — PARTIAL с описанными ограничениями, а
        # не молчаливый IMPLEMENTED (§54 ТЗ): для LDA (svwn) обученных
        # параметров не существует, и запрос отклоняется.
        if correction is DispersionCorrection.NONE:
            availability = Availability.IMPLEMENTED
            since_version = __version__
            limitations = ()
            notes_key = "capability.note.dispersion_none"
        elif correction in (DispersionCorrection.D3_BJ, DispersionCorrection.D3_ZERO):
            availability = Availability.PARTIAL
            since_version = __version__
            limitations = (
                "Область применения: элементы H, B, C, N, O, F, Si, P, S, "
                "Cl, Br, I; функционалы hf, pbe, pbe0, blyp, b3lyp.",
                "Для LDA (svwn) обученных параметров D3 не существует — "
                "такой запрос отклоняется, а не приближается.",
            )
            notes_key = "capability.note.dispersion_d3"
        elif correction is DispersionCorrection.D4:
            availability = Availability.PARTIAL
            since_version = __version__
            limitations = (
                "Область применения: элементы H, B, C, N, O, F, Si, P, S, "
                "Cl, Br, I (поддержка домена); сама модель D4 покрывает "
                "элементы 1..103 и 118 обученных функционалов dftd4 v4.0.1, "
                "включая hf.",
                "Для LDA (svwn) обученных параметров D4 не существует — "
                "такой запрос отклоняется, а не приближается.",
            )
            notes_key = "capability.note.dispersion_d4"
        else:
            # Защита от будущих членов enum: новый член без явной ветки не
            # сможет молча пройти как IMPLEMENTED.
            availability = Availability.NOT_IMPLEMENTED  # type: ignore[unreachable]
            since_version = None
            limitations = ()
            notes_key = "capability.note.not_implemented"
        capabilities.append(
            Capability(
                id=f"dispersion:{correction.value}",
                kind=CapabilityKind.DISPERSION,
                name=correction.value,
                availability=availability,
                since_version=since_version,
                limitations=limitations,
                notes_key=notes_key,
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
                    Availability.NOT_IMPLEMENTED
                    if not available
                    else Availability.PARTIAL
                    if name in _PARTIAL_SCF_OPTIONS
                    else Availability.IMPLEMENTED
                ),
                since_version=__version__ if available else None,
                limitations=(note,) if (not available or name in _PARTIAL_SCF_OPTIONS) else (),
            )
        )

    for name, availability, limitations in _JOB_OPTIONS:
        capabilities.append(
            Capability(
                id=f"job:{name}",
                kind=CapabilityKind.JOB,
                name=name,
                availability=availability,
                since_version=(
                    __version__ if availability is not Availability.NOT_IMPLEMENTED else None
                ),
                limitations=limitations,
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
                limitations=(note,) if note else (),
            )
        )

    return CapabilityRegistry(capabilities)
