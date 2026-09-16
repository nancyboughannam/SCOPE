# SCOPE — Stability-Constrained Optimization for Predictable Edge

Code and experiments for:

> N. Boughannam, A. Nasser, C. Zaki, A. Ramadan, S. Hamrioui, P. Lorenz,
> **"Stability-Aware Multi-Timescale Orchestration for IoE Edge–Cloud Systems."**

SCOPE is a multi-timescale orchestration framework for IoE edge-cloud
systems. It combines fast, per-task rule-based steering at the edge
(inherited from the authors' earlier RAIDER-TRAX framework) with a slower,
cloud-side global planner that periodically re-scores a fixed catalogue of
candidate configurations using machine-learned performance predictors, and
only ever accepts a configuration change if its predicted reconfiguration
cost stays under a hard budget. The goal is to get the benefit of global
optimization without the instability ("configuration churn") that comes
from re-optimizing greedily on every observation.

**If you only want to verify the paper's numbers without running
anything:** every table in the paper has a pre-computed result file already
checked into this repository. Jump to
["Where each table in the paper comes from"](#where-each-table-in-the-paper-comes-from) below —
you can diff those files against the paper's tables directly.

## How this repository is organized

The system has two halves that run as separate processes and talk to each
other over a local HTTP API — they are never compiled or linked together.

```
SCOPE/
├── ThreeBrains/     Java. A modified fork of EdgeCloudSim, the discrete-event
│                    simulator. Runs the vehicular ITS scenario, moves
│                    vehicles, generates tasks, and calls out to the Python
│                    services below to decide where each task goes.
│                    GPLv3-licensed (see NOTICE.md).
│
└── FLaskAPIs/       Python. Three small Flask services (the "global
                     planner" from the paper's Fig. 2) plus the offline
                     scripts used to build the XGBoost predictors and to
                     turn raw simulation output into the paper's tables and
                     figures. MIT-licensed.
```

### Terminology map: paper names ↔ code names

**Read this before anything else.** The paper's method is called **SCOPE**
throughout. Internally, the same thing is identified as **`RT_OPT_ML`** —
an earlier working name from before the paper settled on "SCOPE" as the
framework's name. The two refer to the same system; here's how every name
in the paper matches its counterpart in the code:

| What the paper calls it | What the code calls it | Where it lives |
|---|---|---|
| SCOPE (the full framework) | `RT_OPT_ML` (Java orchestrator policy name), served by `ml_optimizer_service.py` | `FLaskAPIs/ml_optimizer_service.py`, port 5002 |
| RAIDER-TRAX (baseline) | `RAIDER_TRAX` | Heuristic logic lives directly in `VehicularEdgeOrchestrator.java`; SCOPE's fast/edge steering (Algorithm 2 in the paper) extends this same code path |
| RT-OPT (baseline, non-predictive) | `RT_OPT` | `FLaskAPIs/Optimizer.py`, port 5001 |
| RANDOM (baseline) | `RANDOM` | Handled directly in `VehicularEdgeOrchestrator.java`, no external service call |
| EXP3-MC-adapted (baseline) | `EXP3_MC` | `FLaskAPIs/exp3_mc_service.py`, port 5003 |
| SoftCost-OPT (ablation variant) | `SOFTCOST_OPT` | Same service as SCOPE (`ml_optimizer_service.py`), with the hard `R_max` constraint disabled and only the λ-weighted soft penalty active |
| Performance-only (ablation variant) | `PERFORMANCE_ONLY` | Same service as SCOPE, with λ = 0 and the hard constraint disabled |
| — (training-data generation, not in the paper's comparisons) | `SCOPE_DATA_COLLECTION` | Drives the exploration policy that produced the ~15,900-row raw candidate dataset used to train the XGBoost models |

`FLaskAPIs/scope_statistics.py` confirms this mapping directly — its
`--scope` argument defaults to the literal string `RT_OPT_ML`.

## Setup

### Prerequisites

- **Java 21+** to build and run the simulator (`ThreeBrains/`)
- **Python 3.10+** for the optimization services and analysis scripts
- The Java dependencies (CloudSim, Weka, jFuzzyLogic, etc.) are already
  vendored as `.jar` files under `ThreeBrains/lib/` — no separate download
  needed

### Python environment

```bash
cd FLaskAPIs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `xgboost==3.1.2`, `pandas==2.3.3`, and
`scikit-learn==1.8.0` — these three are confirmed to match what was
actually used to train the checked-in models
(`scope_candidate_xgboost_results/model_manifest.json` records the exact
versions). The rest of the dependencies (`flask`, `ortools`, `scipy`,
`matplotlib`, `numpy`) are listed with minimum versions inferred from the
imports; they have not been individually version-locked against the
original run, so if you hit a compatibility issue, that's the first place
to look.

### Compiling and running the simulator

```bash
cd ThreeBrains/scripts/sample_app5
./compile.sh          # builds into ThreeBrains/bin
```

Which policies the simulator exercises is controlled by one line in
`ThreeBrains/scripts/sample_app5/config/default_config.properties`:

```properties
# Main scalability comparison (Fig. 4, Fig. 5, Table VI):
orchestrator_policies= RANDOM,RAIDER_TRAX,RT_OPT_ML,EXP3_MC

# Ablation study (Table VII) - comment out the line above and
# uncomment this one instead:
# orchestrator_policies=SCOPE_DATA_COLLECTION,SOFTCOST_OPT,PERFORMANCE_ONLY,RT_OPT
```

This switch is manual today — there's no flag or separate config file for
it, you edit this line by hand depending on which experiment you're
running. That's a rough edge worth fixing if this repo gets more outside
users; flagging it here so it's not a surprise.

Run a single iteration with:

```bash
./runner.sh <output_dir> default_config edge_devices.xml applications.xml <iteration_number>
```

or use `run_scenarios.sh <parallel_processes> <iterations>` to run several
iterations in parallel (see the script for usage). Each run needs the
relevant Flask service(s) already running and reachable at
`127.0.0.1:5001`/`5002`/`5003` (below).

### Starting the optimization services

Each service is a separate Flask process. Run whichever ones the active
`orchestrator_policies` line needs:

```bash
cd FLaskAPIs

# RT-OPT baseline (also backs RAIDER_TRAX and RANDOM's calls) - port 5001
python3 Optimizer.py

# SCOPE / RT_OPT_ML, SoftCost-OPT, Performance-only - port 5002
# (which variant runs is selected by the "policy" field in each request,
# driven by the simulator's active orchestrator_policies line above)
SCOPE_LAMBDA=0.20 SCOPE_R_MAX=0.15 SCOPE_W_FAILURE=0.50 SCOPE_W_SERVICE=0.50 \
  python3 ml_optimizer_service.py

# EXP3-MC-adapted baseline - port 5003
python3 exp3_mc_service.py
```

`SCOPE_LAMBDA=0.20` and `SCOPE_R_MAX=0.15` are the paper's reported
operating point (Section V-D). **The script's own built-in defaults if you
omit these env vars are different (`LAMBDA=0.50`, `R_MAX=0.20`)** — those
built-in defaults were left over from earlier sensitivity sweeps, not the
paper's final settings. Always set them explicitly as shown above (or use
`FLaskAPIs/catalogue_sensitivity/start_catalogue_service.sh` as a template —
it sets these correctly).

## Where each table in the paper comes from

Every one of these has already been run once, and the output is checked
into this repository — you can inspect or diff the files below without
running anything. The "regenerate" command shows how to reproduce it from
scratch. For some of these, that additionally requires EdgeCloudSim's raw
performance logs, which live in `ThreeBrains/sim_results/` (see the note at
the end of this section for which parts of that folder are, and aren't,
checked in).

| Paper artifact | Already-computed result in this repo | Script that produces it |
|---|---|---|
| Table IV (XGBoost prediction accuracy) | `FLaskAPIs/scope_candidate_xgboost_results/model_metrics.csv` and `model_manifest.json` | `python3 train_candidate_xgboost.py` (reads `scope_candidate_dataset/candidate_dataset_{train,validation,test}.csv`, built by `prepare_candidate_dataset.py`) |
| Table V (λ sensitivity) | `FLaskAPIs/lambda_table_output/lambda_sensitivity_table.tex` and `lambda_sensitivity_summary.csv` | `python3 lambda_sensitivity_table.py --decision-input "Lambda testing_Flask.zip" --results-input "Lambda testing_Simresults.zip" --output-dir lambda_table_output --selected-lambda 0.20` (run from `FLaskAPIs/`; both inputs are checked in as zips, no extraction needed) |
| Table VI (paired baseline comparison, Wilcoxon + bootstrap CIs) | `FLaskAPIs/scope_statistics_results/` (generated on run; not pre-populated in this repo — see note below) | `python3 scope_statistics.py --input <path to your sim_results>` |
| Table VII (ablation: SCOPE vs. SoftCost-OPT vs. Performance-only vs. RT-OPT) | `FLaskAPIs/ablation_figures_final/table1_ablation.tex` | `python3 plot_ablation_results.py` — reads `ablation_results_final/decision_logs/` (checked in) plus raw simulator logs; extract `ThreeBrains/sim_results/Ablation test.zip` into `FLaskAPIs/ablation_results_final/sim_results/` first |
| Table VIII (candidate-catalogue size sensitivity) | `FLaskAPIs/catalogue_runs/pilot_{full,half,quarter}_*` (raw decision logs; the paper's summary table itself is not pre-built in this repo) | `python3 catalogue_sensitivity/catalogue_results_table.py --sim-root ... --decision-root FLaskAPIs/catalogue_runs --output ...` |
| Fig. 3 (Δ*T* sensitivity) | Not pre-built in this repo | Requires re-running the simulator at ΔT ∈ {15, 30, 60, 90}s. This isn't a config-file setting — ΔT is hardcoded as DELTA_T in ThreeBrains/src/edu/boun/edgecloudsim/applications/sample_app5/VehicularEdgeOrchestrator.java (line 71, currently 60.0). Edit that constant, re-run ./compile.sh, then run the simulator as above — once per ΔT value.|
| Fig. 4 / Fig. 5 (scalability curves, service time, VM utilization) | `ThreeBrains/sim_results/full study - 10 iterations/ite1` … `ite10` (raw EdgeCloudSim logs for the full 100–1,800 vehicle sweep, all 4 main-comparison policies, 10 iterations) | The standard EdgeCloudSim MATLAB plotters in `ThreeBrains/scripts/sample_app5/matlab/` (`plotAvgFailedTask.m`, `plotAvgServiceTime.m`, `plotAvgVmUtilization.m`) read this log format directly. We have not confirmed these produce pixel-identical output to the paper's figures (they may have been adapted), but they're the standard tool for this data and a reasonable starting point. |

**Important caveat on regenerating from scratch:** `plot_ablation_results.py`,
`scope_statistics.py`, and `catalogue_results_table.py` all combine two
things: the optimizer's own decision logs (checked into this repo, e.g.
`ablation_results_final/decision_logs/`) and EdgeCloudSim's raw
`*_GENERIC.log` performance output. That raw output lives under
`ThreeBrains/sim_results/`:

- **Main scalability comparison (Table VI, Fig. 4, Fig. 5):** the full raw
  sweep — 10 iterations × all 4 main-comparison policies × 100–1,800
  vehicles — is checked in under
  `ThreeBrains/sim_results/full study - 10 iterations/`.
- **Ablation study (Table VII):** its raw simulator logs are checked in
  as `ThreeBrains/sim_results/Ablation test.zip` — extract this into
  `FLaskAPIs/ablation_results_final/sim_results/` before running
  `plot_ablation_results.py`; that's where its `SIM_RESULTS_DIR` looks by
  default. The matching decision logs are already checked in at
  `FLaskAPIs/ablation_results_final/decision_logs/`.
- **λ sensitivity (Table V):** fully reproducible as-is — both required
  inputs (`Lambda testing_Flask.zip` and `Lambda testing_Simresults.zip`)
  are checked into `FLaskAPIs/`, and `lambda_sensitivity_table.py` reads
  zip archives directly, no extraction needed.
- **R_max sensitivity:** discussed in the paper (Section V-D-2) in prose
  only — there is no dedicated table for it, so there's no script here to
  reproduce one. Supporting raw data is checked in at
  `ThreeBrains/sim_results/Rmax testing.zip` and
  `FLaskAPIs/rmax_test/decision_logs/` for anyone who wants to look at it
  directly.

If any of the above is still missing for a given table, regenerating it
from absolute scratch means actually running the simulator sweep yourself,
which — 10 seeded iterations × up to 4 policies × up to 18 vehicle counts —
is a real compute job, not a quick script run.

Three scripts (`plot_ablation_results.py`, `scope_statistics.py`,
`catalogue_sensitivity/catalogue_results_table.py`) previously had the
original author's personal machine paths (`/Users/nancyboughannam/...`)
hardcoded as defaults. These have been changed to repository-relative
defaults, overridable with environment variables — see each script's
top-of-file comments.

## Known limitations

This section says the same things directly rather than leaving you to
discover them:

- **The paper's methods section states train/validation/test was split by
  hand-picking iterations 1, 2, 4 / 3 / 5.** The code (`prepare_candidate_dataset.py`,
  `split_run_ids()`) actually does a seeded random shuffle of iteration IDs
  and takes the resulting assignment — it isn't a manual choice, it's a
  deterministic-but-automatic one. The committed dataset summary
  (`scope_candidate_dataset/candidate_dataset_summary.json`) confirms the
  shuffle landed on exactly that assignment. Worth knowing if you're
  trying to reproduce this split with a different seed and get a different
  outcome — that's expected, not a bug.
- **The `orchestrator_policies` switch between the main comparison and the
  ablation study is a manual config edit**, not a command-line flag (see
  above).
- **This repo is large to clone**, deliberately — the raw simulator output
  under `ThreeBrains/sim_results/` (split into per-iteration folders to
  stay under GitHub's 100MB per-file limit) plus the various result CSVs
  and vendored `.jar` dependencies add up to several hundred MB. That's a
  conscious trade-off for reproducibility, not an oversight: everything
  needed to verify the paper's numbers is a `git clone` away, no external
  hosting account required. If clone size ever becomes a real problem,
  Git LFS or an external host (e.g. Zenodo, which would also get you a
  citable DOI) is the way out — that would require rewriting git history,
  so it hasn't been done here.
- **Licensing**: see `NOTICE.md`. Short version — this repo mixes MIT
  (SCOPE's own code) and GPLv3 (the vendored `ThreeBrains` simulator fork)
  under one roof, which is legally fine because the two only talk to each
  other over HTTP, but it does mean the two directories are not
  interchangeable if you plan to reuse or redistribute parts of this code.
- **This evaluation is entirely simulation-based** (EdgeCloudSim, a
  synthetic vehicular ITS scenario). As the paper's own conclusion states,
  the candidate-performance predictors are trained and evaluated on
  simulator-generated data; how they'd hold up on real telemetry, with
  noisy or delayed measurements, is explicitly future work, not something
  this repository demonstrates.

## Paper status

*"Stability-Aware Multi-Timescale Orchestration for IoE Edge–Cloud Systems"*
is currently unpublished and under submission. A formal citation (venue,
year, DOI) will be added here once it's accepted — for now, this repository
is the reference if you want to link to the work in progress.

## License

MIT for everything except `ThreeBrains/`, which stays GPLv3. See `LICENSE`
and `NOTICE.md` for the full explanation and what it means if you reuse
this code.

## Contact

Nancy Boughannam — University of Haute Alsace, Mulhouse, France —
boughannamnancy@gmail.com
