
## PBC (TPSS-c) forensics — RESOLVED + DECISION (2026-09-01)

### Mechanism (source + binary confirmed)
- .so = libxc "7.0.0" build (pyscf 2.14.0 wheel; CMake pins gitlab 7.0.0 tarball; disasm shows
  maple2c-era code: calls log/sqrt/cbrt/exp only; func_info flags 0x1008f = 3D|NEEDS_TAU|HAVE_ALL,
  NO ENFORCE_FHC bit — yet behavior clamps: the 7.0.0-archive build clearly carries a
  driver-level sigma clamp `my_sigma = min(sigma, 8.0*rho*tau)` (ENFORCE_FHC semantics: value
  clamped, clamp derivative NOT propagated). Flag paradox unresolved but behavior is settled.
- A/B split in clamped region: EXACTLY t68 = ((sigma_eff*(1/rho))*(1/tau))/8 > 1 with
  sigma_eff = 8.0*(rho*tau) — 183/183 ground-truth grid points (vt==0 ⟺ B).
- In clamped region VXC depends on (rho, tau) ONLY (100x sigma variation -> bit-identical).
- A mode: unclamped partials at sigma_eff — machine precision.
- B mode: vs = rho*F_t2(t2e,1)*rho^(-8/3) exact; vt = 0 exact;
  vr = rho*(F_t2*a_B + F_rho), a_B(rho,tau) a 2-D function, NOT any closed chain rule
  (not c2, not c2+phys, not raw-aux, not f(t2e,rs), not f(w)). NO consistent formula exists.
- libxc upstream MR !800 documents this exact defect: "The Fermi-hole-curvature guard corrupts
  mGGA derivatives ... the clamp's derivative is never propagated ... energy exactly right,
  derivatives wrong ... fix deferred to the post-7.1 refactor." B mode = that known defect.

### DECISION (engineering)
- Implement TPSS-x and PBC-c as the UNCLAMPED kernels (raw sigma, u = sigma/(8*rho*tau) raw):
  * matches oracle at machine precision for ALL physical (multi-electron RKS, FHC-valid) inputs
    (there the driver clamp is inactive: sigma < 8*rho*tau with margin -> oracle == unclamped);
  * self-consistent (CN check passes);
  * for unphysical sigma > 8*rho*tau (single-orbital densities incl. H atom tail, noise):
    oracle clamps sigma and its B-mode derivatives are the known libxc defect; we do NOT
    replicate it. Documented in functional.py docstring + ADR note.
- H atom (N=1: 8*rho*tau == |grad rho|^2 exactly) sits ON the coin-flip boundary -> oracle
  PBC derivatives there are a dense A/B flip; cross-check tolerance for H must be relaxed or
  H excluded from PBC-c derivative cross-check (use energy only). H2O/CH4/NH3 (N>1) are clean.

### M1 integration (START NOW)
- functional.py: insert TpssExchange / PbcCorrelation / Tpssh before FUNCTIONALS dict (~1846)
  (module currently NameErrors at import: dict references undefined Tpssh).
- PbcCorrelation unclamped: e = f0*(1+2.8*f0*u^3), f0 = (1+0.53u^2)*g - 1.53u^2*br,
  g = f_pbe_c(rs,0,t2), br = 0.5*max(f_pbe_c(rs2^(1/3),+1,2^(2/3)*t2), g) + 0.5*max(...,-1,...),
  t2 = sigma*rho^(-8/3), u = sigma/(8*rho*tau) RAW, rs = (3/4pi*rho)^(1/3).
  Derivatives: analytic (sympy at import, or chain rule from PBE-c (F_rs, F_t2) partials).
  Verified energy formula: tools/scratch_pbc_final.py (worst 1.23e-12 vs oracle, clamped region).
- dft.py: xc_matrix_and_energy tau term 0.5*einsum("p,pgd,phd->gh", w*vtau, gradients, gradients);
  run_rks tau_at_points; gradients.py requires_tau guard; run_uks raise (M4).
- registry mgga -> PARTIAL; tests vs PySCF (H2O/CH4/NH3); mypy+ruff.

## TPSS-x mystery RESOLVED (2026-09-02) — .so = libxc 7.0.0, vanilla TPSS params
- Oracle kernel = maple source `maple/tpss_x.mpl` (TAGGED 7.0.0), NOT the stale
  generated C in the 7.0.0 archive (src/maple2c/mgga_exc/mgga_x_tpss.c):
  * archive C: qb sqrt = 9(1 + b(alpha-1)^2)  -> worst rel 4.6e-3 vs oracle
  * tagged .mpl / .so: qb sqrt = 9(1 + b*alpha*(alpha-1)) -> worst rel 4.3e-16
  (alpha = 1 + (t - S/8)/K_FACTOR_C, K_FACTOR_C=(3/10)(6 pi^2)^(2/3)=4.557790,
   t=2^(2/3) tau rho^(-5/3), S=2^(2/3) sigma rho^(-8/3), b=0.40)
- C e-notation trap: `0.27e2/0.2e2` = 27/20 = 1.35 (NOT 27/2 = 13.5!)
- unpol output doubling: C kernel computes ONE spin channel (t142), driver
  returns tzk0 = 2*t142. eps_pp = -(3/4)(3/pi)^(1/3) rho^(1/3) F.
- Vanilla params {b=0.40, c=1.59096, e=1.537, kappa=0.8040, mu=0.21951,
  BLOC_a=2.0, BLOC_b=0.0} are CORRECT (the .so values-table reads were garbage
  — pointers not mappable; "non-vanilla .so" conclusion was WRONG).
- PBC (2001) correlation = MGGA_C_TPSS in libxc 7.0.0 (no MGGA_C_PBC name).
  Kernel verified at machine precision (worst 1.56e-8 rel = 1e-17 abs on dtau).
- FINAL VERIFICATION (500 physical pts, eps + dE/drho/sigma/tau):
  TPSS-x worst rel 4.8e-13, PBC/TPSS-c worst rel 1.6e-8 (roundoff). DONE.
- tools/scratch_mgga_derive.py: build_tpssx/build_pbc NOW CORRECT (maple form,
  27/20, x2 unpol, real symbols in verify()). Re-runnable.
- TPSSH = 0.25 HF + 0.75 TPSS-x + PBC-c (MGGA_C_TPSS). M1 target.
- NEXT: production classes in functional.py (TpssExchange, PbcCorrelation,
  Tpssh) + dft.py tau plumbing + registry mgga label + RKS tests vs PySCF.
