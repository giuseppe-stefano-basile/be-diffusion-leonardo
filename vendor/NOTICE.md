# Numerical source provenance

The Hummer bounded sampler, original observable/likelihood definitions, and
quad/384-bit C++ propagator are copied from the existing project. The native
Stochkin FPE module and spline model are copied byte for byte from its archived
analysis sources. Counting, reconstruction and convergence functions are
extracted unchanged; `provenance.json` records source and exported hashes.

`hummer_high_precision.py` replaces the platform-specific compiler/library loader.
The compiled C++ arithmetic is unchanged. Its cache lives outside the repository;
MPFR include/library locations are configurable. Arithmetic failures abort fitting.

The native Stochkin integral receives only a concrete float64 dtype specialization
for Numba, with fastmath disabled. Regression tests compare it to the original
function and all eight frozen baseline rates. The FES is held fixed and unsmoothed.

These are project research sources delivered privately for this campaign. No new
redistribution license or ownership claim is introduced by this handoff.
