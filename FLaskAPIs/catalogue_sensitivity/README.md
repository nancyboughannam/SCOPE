# Catalogue-size sensitivity: first pilot

This package adds catalogue-size settings to your uploaded optimizer. Keep your
original service as a backup. No Java source edits are needed for selection.

## 1. Place these files beside your original service

Copy ml_optimizer_service_catalogue.py, catalogue_subset.py and
start_catalogue_service.sh into the directory containing ml_optimizer_service.py.
Keep the existing model directory and candidate dataset there. The new service
uses the same Python dependencies as the original. It does not retrain models.

## 2. Configure a separate simulator experiment

Use a COPY of your simulator configuration, selecting only RT_OPT_ML, with minimum
and maximum vehicles both 1800. Keep the time interval, warmup, applications,
network, mobility and all other settings identical to the original experiment.
Do not run other policies in the catalogue experiment.

The supplied archive contains Java files but no .properties files, classpath/build
settings or launch script. Therefore exact edits to your local simulator launcher
cannot be verified here. Do not guess property names: attach your active
configuration and launch script for those final edits.

Your VehicularMainApp accepts FIVE arguments, in this order:
configFile edgeDevicesFile applicationsFile outputFolder iterationNumber.
It uses simulation seed 20260828 + iterationNumber. Always pass iterationNumber;
its no-argument default is currently 6. For the first pilot, use iteration 1
(seed 20260829) for all three catalogue settings. Use distinct output folders:

catalogue_runs/pilot_full_s1/sim/ite1
catalogue_runs/pilot_half_s1/sim/ite1
catalogue_runs/pilot_quarter_s1/sim/ite1

These simulator folders are separate from the service's decision-log roots.
Creating the Python log root does NOT redirect Java simulation output.

## 3. Run the full-catalogue pilot

Stop the original service on port 5002. In its directory run:

```bash
bash start_catalogue_service.sh 1.0 1 catalogue_runs/pilot_full_s1/decisions
```

In another terminal check:

```bash
curl http://127.0.0.1:5002/health
```

Verify lambda=0.20, r_max=0.15, hard_stability_enabled=true and
catalogue_fraction=1.0. The full count should match your actual dataset
(expected 2536, but the service reports what it actually loads).
Run your simulator once using the separate configuration, full output folder,
and iteration 1. Wait for simulation completion before stopping the service.

## 4. Repeat with half and quarter catalogues

Stop the service, then start the half setting:

```bash
bash start_catalogue_service.sh 0.5 1 catalogue_runs/pilot_half_s1/decisions
```

Check /health, run simulator iteration 1 into the half output folder, wait for it
to finish, then stop the service. Repeat for the quarter setting:

```bash
bash start_catalogue_service.sh 0.25 1 catalogue_runs/pilot_quarter_s1/decisions
```

Each root must be new. Do not rerun the same iteration into existing output/logs.
If a pilot fails, choose a new root for the retry. Do not run multiple simulation
processes at once against this service when comparing planning time.

## 5. What to send back

Send the three simulation ite1 folders and the three decision roots, including
selected_catalogue.csv, catalogue_metadata.json and decision_logs. Check for
optimizer request failures in the simulator console: fallback runs must not be
silently counted as successful SCOPE runs.

After validating the pilot, run iterations 1..10 per setting in fresh FINAL
folders (30 simulations for one subset realization, if full runs are repeated).
The service can stay running for all ten iterations of one setting. Change only
iteration and simulator output folder between runs. Keep subset seed 1 FIXED
across those ten runs: it is different from the simulation seed.

For a stronger subset-composition check, repeat the quarter/half settings with
subset seeds 2 and 3 using the same simulation seeds (40 additional simulations).
The full catalogue does not change with subset seed, so need not be duplicated
for that check. Do not treat multiple subsets sharing a simulation seed as
independent simulation seeds in a pooled statistical test.

## Design and interpretation

For a 2536-row catalogue: quarter=634, half=1268, full=2536 catalogue entries.
The active configuration is always added by the original service. The evaluated
candidate batch can therefore be one larger than the catalogue if the active
configuration is absent. Both counts are logged. Initial values from the Java
code are alpha=0.6, load weights=(0.5,0.7,0.4), thresholds=(85,80,90).
If this initial configuration occurs in the catalogue, it is kept in all subsets.

Subsets use a seed-dependent start and greedy farthest-point selection with
normalized L1 distance across all seven parameters (the original churn scaling).
This aims for parameter-space diversity; it does not guarantee equal feasible
neighborhood coverage or performance. Small is nested within half for a fixed
subset seed. No outcomes or prediction scores are used to select subset members.
The full catalogue preserves its original row order exactly; subsets preserve
original relative row order so existing tie-breaking is retained.

Compare failure percentage, ordinary service time in seconds, number of changed
configurations, sum/max churn, budget violations and optimizer_time_ms. Include
feasible_candidate_count to help explain differences. Existing service timing
covers per-request planning, not HTTP round-trip or startup subset construction.
Use the same hardware and execution conditions for timing. The service already
logs these decision metrics; catalogue metadata is now added to each row and
/health. Full service integration cannot be tested here because the trained
models, dataset and runnable simulator build are not in the attachments.

The objective, feasibility constraint, predictions, response format and Java
policy logic have not been changed. These experiments test sensitivity to
REDUCING the existing catalogue, not expansion beyond its support. A smaller
catalogue need not give worse results: predictions and later simulation dynamics
can make results non-monotonic. Report the observed pattern without assuming it.

Existing full-catalogue results can be reused for performance only if all settings,
models, data and seeds match. For timing, a matched new full-catalogue run is
preferable. The original service defaults were lambda=0.50 and r_max=0.20;
the launcher explicitly sets the study point lambda=0.20 and r_max=0.15.

## Validation performed on this package

Checked nested membership, initial-configuration retention, full-catalogue row
identity, membership reproducibility after input row reordering, alternative
subset seeds, and unchanged ASTs for candidate batching, objective scoring,
churn computation and Java response formatting. Real pilot runs are still needed.
