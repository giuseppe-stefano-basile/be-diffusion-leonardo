# Be diffusion pilot on Leonardo Booster

Standalone handoff for `gbasile0`, allocation `INF26_biophys`. This campaign
collects new unbiased diffusion data for both R25 and R27 and compares the
resulting provisional threshold-passage rates with the corrected-FES baseline.

The pilot has 60 structural parents from **60 distinct original source-block
groups**, four 2 ps releases each, plus 24 matched preparation-control releases.
Every trajectory records both logistic Be–O CVs and coordinates every 0.005 ps.
Nominal total MD, including preparation and controls: **1.056 ns aggregate**;
bounded preparation extensions permit at most **1.320 ns**. Source blocks are
clustering labels, not proven independent samples. No MD runs on the source PC.

## 1. Clone and stage inputs

Once published:

```bash
git clone https://github.com/giuseppe-stefano-basile/be-diffusion-leonardo.git
cd be-diffusion-leonardo
git checkout v1.0.0
```

The supplied code archive is an equivalent route before GitHub publication.
Authenticate using your usual private GitHub access; credentials do not belong
in this repository. Obtain `be-diffusion-inputs-v1.0.0.tar.gz` from the handoff
artifacts and transfer it once to Leonardo, for example into `$HOME`.
Models, baseline time series/posteriors, and trajectory outputs are outside Git.

Select the `INF26_biophys` project with your normal CINECA project setup so
`$WORK` refers to that allocation, then:

```bash
export BE_INPUTS="$WORK/be-diffusion/inputs-v1"
export BE_RUN_DIR="$WORK/be-diffusion/pilot-v1"
export BE_RUNTIME_ENV="$HOME/be-diffusion-runtime.env"
cp config/runtime.env.example "$BE_RUNTIME_ENV"
```

Edit `$BE_RUNTIME_ENV` to reproduce **your existing working** MPI/CUDA and
Python activation and exact MPI-enabled PLUMED executable/kernel. The LAMMPS
default is `/leonardo/home/userexternal/gbasile0/lammps-glob-dev/build-MOL/lmp`.
No other user’s Python environment is assumed and no installation is modified.
The scripts require Python 3.10+ and NumPy. `preflight --cpu-only` does not
import CUDA or run LAMMPS and can be used on a login node:

```bash
source "$BE_RUNTIME_ENV"
python scripts/stage_inputs.py "$HOME/be-diffusion-inputs-v1.0.0.tar.gz" --destination "$BE_INPUTS"
python -m be_pilot preflight --cpu-only
```

The archive extractor checks the configured archive SHA-256, rejects unsafe
members, verifies every external file, and refuses different existing inputs.
The selected structures and full historical source hashes are in `seeds/`.
Historical absolute paths inside archived provenance are descriptive only;
no operational code dereferences them.

## 2. Qualify on four Booster GPUs

```bash
python -m be_pilot smoke --runtime-env "$BE_RUNTIME_ENV"
```

This submits a one-node, four-GPU, ten-minute `boost_qos_dbg` job. Each GPU runs
an isolated 0.05 ps restrained stage and a 0.1 ps unbiased stage. The job validates
all saved coordinates and both CVs. These frames never enter production analysis.

Qualification records binary/kernel/model hashes, MPI linkage, Python packages,
GPU names, raw-model identity, conversion provenance and wall-clock throughput.
The raw checkpoint is exported to ML-IAP once **inside the allocated GPU job**
and reused only with matching package versions and checksums. Converted models
stay under `$BE_RUN_DIR/runtime`. The existing LAMMPS build is preserved; if its
PLUMED linkage cannot be verified, the job stops with an actionable diagnostic.

Wait for the smoke job to finish and inspect its logs, then:

```bash
python -m be_pilot status
test -f "$BE_RUN_DIR/QUALIFIED.json"
python -m be_pilot submit --runtime-env "$BE_RUNTIME_ENV"
```

## 3. Production, recovery and completion

Production requests one Booster node, four GPUs, 32 CPU cores and 24 hours
under the normal `boost_usr_prod` partition. Four workers keep one LAMMPS
process on each allocated GPU, moving through a deterministic parent queue.
No local GPU-id polling or other users' processes are involved.

The six bands are:

| CV | Centers | Half-width | Parents per band |
|---|---|---|---:|
| R25 | 4.6500, 4.7375, 4.8250 | 0.025 CN | 10 |
| R27 | 4.8000, 4.8800, 4.9600 | 0.025 CN | 10 |

Each primary parent gets 8 ps flat-bottom preparation (`KAPPA=20000` in PLUMED
kJ/mol units). A deterministic, uniformly selected in-band frame from the last
2 ps seeds four releases with distinct velocity and thermostat seeds. If there
is no eligible frame, extend preparation in 2 ps segments, up to 4 ps extra,
then quarantine that parent rather than replace it using production outcomes.
The first selected parent in each band also receives a 16 ps preparation branch
continued from the 8 ps branch's final restart, plus four control releases.
Actual preparation lengths and selected frame identities are recorded.

All releases are unbiased: no walls, restraints, or metadynamics. Dynamics keep
the original 376 atoms, box, trained model, 298 K, 0.5 fs timestep, NVE integrator
and Langevin damping 0.5 ps. R25/R27 switches have slope 4 Å⁻¹ and radii
2.5/2.7 Å. Model atom mapping is `Be O H` with fixed Be + 125 `(O H H)` ordering.

```bash
python -m be_pilot status
python -m be_pilot resume --runtime-env "$BE_RUNTIME_ENV"
```

Every stage has immutable `attempt_NNN` folders. A completion marker is written
only after clean LAMMPS termination, complete time/coordinate output, fixed-box
checks, and independent reconstruction of both CVs. Its hashes are checked on
resume. Completed stages are not rerun; failed or interrupted attempts remain
available and receive a new attempt directory on retry. Concurrent execution
against the same run directory is locked out. A Slurm signal stops new stages
and terminates active children; incomplete stages restart from their stage input.
A changed runtime/input fingerprint requires a new smoke-qualified run directory.

A successful pilot contains 240 primary and 24 control releases, or an explicit
quarantine accounting for missing shots. Failures do not count as completed MD.
The smoke completion records provide an initial timing estimate; use completed
production stage timings for a reliable campaign ETA rather than historical speed.

## 4. CPU analysis and comparisons

Use the same Python if it has the analysis requirements, or configure a separate
CPU environment in the activation script for the analysis job. CPU fitting needs
NumPy, SciPy, Matplotlib, Numba, a C++17 compiler, quadmath, and MPFR development
headers/library. Set `HUMMER_MPFR_INCLUDE` / `HUMMER_MPFR_LIBDIR` if the compiler
cannot discover MPFR. Do not change the working GPU environment to install these.

```bash
python -m be_pilot analyze --phase audit
sbatch --export=ALL scripts/analyze.sbatch
```

The audit first reproduces both frozen count matrices and all eight corrected-FES
baseline rate estimates, then writes coverage, directional transition counts,
source-group counts, failed attempts and quarantine information. The analysis
job uses four CPU cores in `lrd_all_serial`, with a four-hour limit and no GPUs.
MCMC chunks checkpoint independently; resubmit the same command to continue
after walltime interruption. Default schedule: four chains, 480000 steps/chain,
60000 initial warm-up steps, thinning 20. Global prior moves and warm-up-only
adaptation retain the existing bounded uniform-linear-rate target.

After the primary fit, run the full sensitivity suite, resubmitting as needed:

```bash
BE_ANALYSIS_VARIANT=suite sbatch --export=ALL scripts/analyze.sbatch
```

The suite includes primary, doubled lag, 0.2 ps initial exclusion, 32/48 bins,
matched 8/16 ps preparations, and five held-out original source-family folds.
Each held-out fit removes the entire family from old and new training data and
exports a posterior-predictive log score for held-out transitions. Preparation
controls are analyzed separately; they are never pooled into primary counts.

Primary counting preserves R25's 0.1 ps frame phase and lag, and R27's 0.005 ps
frames with 0.05 ps lag. Origins are lag-spaced and non-overlapping, with the
original 0.1 ps exclusion and Hummer bin-center shift. Finer storage supplies
validation and sensitivities rather than extra independent observations.
R25's archived 0.1 ps spacing does not support a shorter-lag pooled fit; no
interpolation is used. All descendants retain their original source-group ID.

The pooled likelihood uses old + new counts once, with unchanged `g_narrow`
priors (`|g| <= 10 kBT`, linear forward rates `< 1000 ps⁻¹`), and no added
smoothness penalty. Baseline posterior states initialize fitting only; they
are not used as a new prior. Driving `g/k` and derived `D` convergence are
reported separately from scalar-rate convergence (Rhat ≤1.01, bulk/tail ESS ≥400).
Arithmetic failures abort the fit rather than silently truncate its support.

Kinetics uses the corrected, fixed, unsmoothed FESs, positive-curvature targets
R25 **4.76** / R27 **4.92**, original source cores and reflecting boundaries,
native Stochkin integral, harmonic D reconstruction and division by four.
Both raw-median and smoothed-D sensitivity results propagate every complete
retained D-profile draw, recording failed solves. Point estimates repeat
4001/8001-point grid checks and source-core/boundary sensitivities.

Outputs under `$BE_RUN_DIR/analysis/results/<variant>/` include reports,
convergence diagnostics, association/dissociation comparison plots
(PDF/PNG/SVG), diffusion/FES plots and conditional width ratios to the baseline.
Complete rate arrays and MCMC checkpoints live under `analysis/counts/<CV>/<variant>/`.
Review `analysis/coverage.csv`, matched preparations, and held-out scores before
recommending further MD. No command automatically expands the campaign.

Rates are provisional reciprocal threshold-passage MFPTs per water. Retained-chain
spread conditions on the FES, CV, potential, preparation protocol and priors;
it is not validated uncertainty for completed water exchange. FES and model
uncertainty remain unquantified. Narrower spread alone does not demonstrate
improved physical accuracy or independence of source blocks.

## Tests and provenance

```bash
BE_TEST_INPUTS="$BE_INPUTS" OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests exercise lineage/geometry, periodic CV reconstruction, random-seed
uniqueness, counting conventions, immutable completion, corruption detection,
interruption/retry, unbiased production, and real frozen-input regression.
They use synthetic process output and **do not run MD or fit the real data**.
`SHA256SUMS.json` seals shipped code/configuration/seeds; the external archive
has a separate manifest. A tagged revision and input/runtime fingerprints tie
each run to its exact inputs. Existing workspace scientific results are preserved.

References: [CINECA Leonardo scheduler](https://docs.hpc.cineca.it/hpc/leonardo.html)
and [storage documentation](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html).

## Private GitHub publication or offline cloning

If the GitHub repository has not yet been published, the supplied Git bundle
contains the complete local repository, `main` branch and `v1.0.0` tag:

```bash
git clone /path/to/be-diffusion-leonardo-v1.0.0.bundle be-diffusion-leonardo
cd be-diffusion-leonardo
git checkout v1.0.0
```

With GitHub CLI authenticated as `giuseppe-stefano-basile`, publish from this
directory with `python scripts/publish.py`. An environment-provided `GH_TOKEN`
is also supported. The script verifies package integrity, creates/checks the
private repository, and pushes the commit and tag atomically. Credentials and
external model/baseline files are never committed.
