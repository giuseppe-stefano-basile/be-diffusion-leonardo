# Handoff validation

CPU validation on the recovered workspace passed 24 tests. Both frozen transition
matrices reproduce exactly. All eight raw/smoothed association/dissociation rates
reproduce with maximum relative error 2.22e-16.

A fixture using subsets of existing D profiles exercised 640 posterior MFPT
solves and generated all comparison plots, with zero failed solves. These fixture
plots are validation artifacts, not new scientific pilot results.

Tests include exported geometry/lineage, periodic dual-CV reconstruction, distinct
random seeds, bounded preparation/quarantine, matched preparation continuation,
production bias rejection, corrupted/incomplete output, interrupted retries,
completed-stage protection, scheduler qualification, native arithmetic, and CPU
checkpoint reuse. The quad and 384-bit paths agree on a synthetic target.

No local MD, GPU qualification, model conversion, or real-data diffusion fitting
was run. The actual Leonardo binary, Python environment and PLUMED installation
must pass the supplied four-GPU smoke job before production.
