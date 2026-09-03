# Закреплённые источники: счётчики ЧК (mctc-lib)

Используется в стадии 3 D4 (EEQ-заряды): модель eeq2019 считает ЧК
собственным erf-счётчиком (не erf_dftd4, что применяется к C6).

Источник: **mctc-lib** (grimme-lab/mctc-lib), ветка main, выгружено
2026-08-31. Лицензия Apache-2.0.

| Файл (оригинал → здесь) | Что используется |
| --- | --- |
| `src/mctc/data/covrad.f90` → `covrad.f90` | covalent_rad_2009 (Pyykko/Atsumi 2009; металлы −10%) и covalent_rad_d3 = (4/3)·covalent_rad_2009 — радиусы для ЧК (идентичны таблицам dftd4 `data/covrad.f90`) |
| `src/mctc/ncoord/erf.f90` → `erf_ncoord.f90` | erf-счётчик: ½(1+erf(−kcn·(r−rc)/rc^norm_exp)), rc = rcov_i+rcov_j, norm_exp=1; без EN-фактора (в отличие от erf_dftd4) |
| `src/mctc/ncoord/type.f90` (не закреплён, см. README multicharge) | цикл по парам (пропуск r²<1e-12 и r²>cutoff²; self-пара исключается тем же порогом), `directed_factor=+1`, плавное ограничение ЧК `log_cn_cut` |

Ключевые константы модели (new_eeq2019_model, multicharge param.f90):
cutoff = 25 bohr, kcn = 7.5, cn_max = 8.0, rcov = covalent_rad_d3.

Ограничение ЧК — не min(cn, cnmax), а плавная функция
(cn_max = 8.0): `cn_cut = ln(1+e^cnmax) − ln(1+e^(cnmax−cn))`;
асимптотически → cnmax при cn→∞, → cn при cn<<cnmax.
