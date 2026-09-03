# Закреплённые источники: модель зарядов EEQ (multicharge)

Стадия 3 D4 (EEQ-заряды) реализована по исходникам **multicharge**
(grimme-lab/multicharge, ветка main, выгружено 2026-08-31) и **mctc-lib**
(grimme-lab/mctc-lib, ветка main, выгружено 2026-08-31).

Модель по умолчанию в dftd4 4.x — `d4_qmod%eeq` (multicharge eeq2019):
dftd4/src/dftd4/model/d4.f90 `new_d4_model` → `new_eeq2019_model`.

## multicharge (grimme-lab/multicharge, main)

| Файл | Что используется |
| --- | --- |
| `param/eeq2019.f90` (здесь: `eeq2019.f90`) | поэлементные таблицы eeq_chi, eeq_eta, eeq_kcnchi, eeq_rad (103 элемента, max_elem=103) — **вход генератора** `tools/generate_d4_data.py` |
| `param.f90` | `new_eeq2019_model`: cutoff=25 bohr, cn_exp=7.5, cn_max=8.0, rcov=mctc::get_covalent_rad |
| `model/eeq.f90` | модель: xvec_i = −chi + kcnchi·cn/√(cn+1e-14), xvec(n+1)=Q; матрица A: A_ii=eta+√(2/π)/rad, A_ij=erf(√(r²γ))/r, γ=1/(rad_i²+rad_j²), последняя строка/столбец = 1 (A_{n+1,n+1}=0) |
| `model/type.F90` | `solve`: Bunch-Kaufman (sytrf/sytrs); `local_charge`: qloc=Q/nat (для EEQ в xvec не входит) |
| `charge.f90` | `get_charges`: CN от ncoord модели → local_charge → solve |

Оригинал: https://github.com/grimme-lab/multicharge (Caldeweyher et al.,
J. Chem. Phys. 2019, 150, 154122, DOI 10.1063/1.5090222). Лицензия Apache-2.0.

## mctc-lib (grimme-lab/mctc-lib, main)

| Файл | Что используется |
| --- | --- |
| `data/covrad.f90` (здесь: `covrad.f90`) | covalent_rad_2009 (Pyykko/Atsumi 2009, металлы −10%), covalent_rad_d3 = (4/3)·covalent_rad_2009 — радиусы ЧК |
| `ncoord/erf.f90` (здесь: `erf_ncoord.f90`) | erf-счётчик: ½(1+erf(−kcn·(r−rc)/rc)), rc=rcov_i+rcov_j, без EN-фактора |
| `ncoord/type.f90` | цикл пар (r<1e-12 пропуск, cutoff²), плавное ограничение ЧК: cn_cut = ln(1+e^cnmax) − ln(1+e^(cnmax−cn)) |

Оригинал: https://github.com/grimme-lab/mctc-lib. Лицензия Apache-2.0.
