#!/usr/bin/env python3
"""Extract SCOPE results and compare paired iterations, without simulation reruns.

Install: python3 -m pip install numpy scipy
Run: python3 scope_statistics.py --input /path/to/sim_results
Default: ite1..ite10, 1800 vehicles, ALL_APPS, three external comparisons.
Input may also be a ZIP. Use --iterations 3 for the sample only.
Use --vehicles 100 200 300 ... to explicitly select additional loads.
Use --baselines SOFTCOST_OPT PERFORMANCE_ONLY RT_OPT for separate ablation logs
(identifiers must match filenames). This extracts failure and service metrics;
churn requires separate decision logs and is NOT inferred here.

Outputs: raw_results.csv, paired_differences.csv, statistics.csv, table.md,
table.tex and methods.txt. Existing output files with these names are replaced.
Iteration IDs are pairing keys, NOT verified seed IDs: confirm matching seeds
across policies and independent seeds across iterations from your run setup.

Metric mapping matches supplied plotGenericLine / plotAvgFailedTask /
plotAvgServiceTime: first numeric row, col2/(col1+col2)*100, and col5.
Service time is seconds; this does NOT calculate deadline-normalized time.

Statistics: two-sided exact signed-rank sign randomization (average tied ranks,
zero differences discarded); paired rank-biserial effect size; percentile 95%
bootstrap CI of mean paired difference (50,000 resamples, reproducible RNG).
Positive baseline-minus-SCOPE differences/effects favor SCOPE for both metrics.
Holm correction covers ALL comparisons/metrics/loads/apps in one invocation.
CIs are pointwise, not multiplicity-adjusted, and concern the mean difference;
Wilcoxon tests symmetry about zero, not the mean. Interpretation as a location
shift assumes symmetric paired differences. Independent seed pairs required.
Small-sample bootstrap CIs can be unstable; 3-run outputs are demonstration only.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
"""
import argparse
import csv
import io
import math
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
from scipy.stats import rankdata

# Not checked into the repository - this is EdgeCloudSim's raw simulator
# output (*_GENERIC.log files), produced by actually running the simulator
# (see the README's reproduction section). Point --input at wherever your
# own run's output lives; this default assumes the conventional layout next
# to a sibling ThreeBrains checkout.
DEFAULT_INPUT = str(
    Path(__file__).resolve().parent.parent / "ThreeBrains" / "sim_results" / "full study"
)
NAME = re.compile(r'SIMRESULT_ITS_SCENARIO_(.+)_(\d+)DEVICES_(.+)_GENERIC\.log$')


def signed_rank(d):
    # Round numerical subtraction noise at 12 decimal places, in metric units.
    d = np.round(np.asarray(d, dtype=float), 12)
    nz = d[d != 0]
    if not len(nz):
        return 0.0, 1.0, 0.0, 0
    ranks = rankdata(abs(nz), method='average')
    # Double ranks so tied average ranks become integer weights. Dynamic
    # programming counts every sign assignment exactly, including ties.
    weights = np.rint(2 * ranks).astype(int)
    counts = {0: 1}
    for w in weights:
        nxt = counts.copy()
        for s, count in counts.items():
            nxt[s + int(w)] = nxt.get(s + int(w), 0) + count
        counts = nxt
    total = int(weights.sum())
    positive = int(weights[nz > 0].sum())
    tail = min(positive, total - positive)
    extreme = sum(c for s, c in counts.items() if min(s, total - s) <= tail)
    return tail / 2, extreme / (2 ** len(nz)), (2 * positive - total) / total, len(nz)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default=DEFAULT_INPUT)
    p.add_argument('--output', default='scope_statistics_results')
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument(
        '--vehicles',
        nargs='+',
        type=int,
        default=list(range(100, 1801, 100))
    )
    p.add_argument('--apps', nargs='+', default=['ALL_APPS'])
    p.add_argument('--scope', default='RT_OPT_ML')
    p.add_argument('--baselines', nargs='+', default=['EXP3_MC', 'RAIDER_TRAX', 'RANDOM'])
    p.add_argument('--bootstrap', type=int, default=50000)
    p.add_argument('--rng-seed', type=int, default=20260912)
    a = p.parse_args()
    if a.iterations < 2 or a.bootstrap < 1000:
        p.error('Require at least 2 iterations and 1000 bootstrap resamples.')
    if a.scope in a.baselines or len(set(a.baselines)) != len(a.baselines):
        p.error('Baselines must be distinct and exclude SCOPE.')
    if len(set(a.vehicles)) != len(a.vehicles) or len(set(a.apps)) != len(a.apps):
        p.error('Duplicate vehicle counts or apps.')
    source = Path(a.input).expanduser()
    if not source.exists():
        p.error(f'Input does not exist: {source}')
    z = zipfile.ZipFile(source) if source.is_file() else None
    names = z.namelist() if z else [str(x.relative_to(source)) for x in source.rglob('*_GENERIC.log')]
    selected = {}
    policies = [a.scope] + a.baselines
    for name in names:
        parts = Path(name).parts
        if '__MACOSX' in parts:
            continue
        its = [re.fullmatch(r'ite(\d+)', x) for x in parts[:-1]]
        its = [int(x.group(1)) for x in its if x]
        m = NAME.fullmatch(parts[-1])
        if len(its) != 1 or not m:
            continue
        policy, vehicle, app = m.group(1), int(m.group(2)), m.group(3)
        iteration = its[0]
        if policy not in policies or vehicle not in a.vehicles or app not in a.apps or not 1 <= iteration <= a.iterations:
            continue
        key = (iteration, policy, vehicle, app)
        if key in selected:
            raise ValueError(f'Duplicate result for {key}: {selected[key]} and {name}')
        selected[key] = name
    raw, lookup, missing = [], {}, []
    for v in a.vehicles:
        for app in a.apps:
            for policy in policies:
                for it in range(1, a.iterations + 1):
                    key = (it, policy, v, app)
                    if key not in selected:
                        missing.append(str(key)); continue
                    name = selected[key]
                    content = z.read(name).decode('utf-8-sig') if z else (source / name).read_text(encoding='utf-8-sig')
                    lines = content.splitlines()
                    if len(lines) < 2 or not lines[0].startswith('#'):
                        raise ValueError(f'Unexpected log header: {name}')
                    values = [float(x) for x in lines[1].split(';')]
                    completed, failed, service = values[0], values[1], values[4]
                    if not all(math.isfinite(x) and x >= 0 for x in (completed, failed, service)) or completed + failed <= 0:
                        raise ValueError(f'Invalid task counts/service time: {name}')
                    if completed == 0:
                        raise ValueError(f'No completed tasks; service-time comparison undefined: {name}')
                    row = dict(iteration=it, policy=policy, vehicles=v, app=app,
                               completed_tasks=completed, failed_tasks=failed,
                               failure_rate_pct=100*failed/(completed+failed), service_time_s=service, source=name)
                    raw.append(row); lookup[key] = row
    if z: z.close()
    if missing:
        raise ValueError(f'Missing {len(missing)} required logs; no partial analysis produced. First missing entries:\n' + '\n'.join(missing[:15]))
    rng = np.random.default_rng(a.rng_seed)
    stats, pairs = [], []
    for v in a.vehicles:
        for app in a.apps:
            for b in a.baselines:
                for metric in ['failure_rate_pct', 'service_time_s']:
                    x = np.array([lookup[(i,a.scope,v,app)][metric] for i in range(1,a.iterations+1)])
                    y = np.array([lookup[(i,b,v,app)][metric] for i in range(1,a.iterations+1)])
                    d = y-x
                    for i in range(a.iterations):
                        pairs.append(dict(iteration=i+1,vehicles=v,app=app,baseline=b,metric=metric,scope_value=x[i],baseline_value=y[i],baseline_minus_scope=d[i]))
                    w,pv,r,nz = signed_rank(d)
                    means = d[rng.integers(0,len(d),size=(a.bootstrap,len(d)))].mean(axis=1)
                    lo,hi = np.quantile(means,[.025,.975])
                    stats.append(dict(vehicles=v,app=app,baseline=b,metric=metric,n_pairs=len(d),n_nonzero=nz,
                        scope_mean=x.mean(),scope_sd=x.std(ddof=1),baseline_mean=y.mean(),baseline_sd=y.std(ddof=1),
                        mean_improvement=d.mean(),ci95_low=lo,ci95_high=hi,rank_biserial=r,wilcoxon_W=w,p_raw=pv))
    order = sorted(range(len(stats)), key=lambda i: stats[i]['p_raw'])
    adjusted = 0.0
    for j,i in enumerate(order):
        adjusted = max(adjusted,min(1.0,(len(stats)-j)*stats[i]['p_raw']))
        stats[i]['p_holm'] = adjusted
        stats[i]['significant_holm_005'] = adjusted < .05
    out = Path(a.output).expanduser(); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'raw_results.csv',raw); write_csv(out/'paired_differences.csv',pairs); write_csv(out/'statistics.csv',stats)
    note = ('Positive differences and rank-biserial effects favor SCOPE. Failure differences are percentage points; service differences are seconds. '
            '95% CIs are paired percentile-bootstrap intervals for mean differences, not simultaneous intervals. '
            f'Holm adjustment covers all {len(stats)} tests in this invocation. Pairing uses iteration folders; actual seeds must be confirmed from the run setup.\n')
    if a.iterations < 10:
        note = 'SAMPLE ANALYSIS: fewer than 10 iterations; do not present as the final 10-run study.\n\n' + note
    md = [note,'| Vehicles | App | Baseline | Metric | n | Mean improvement [95% CI] | Rank-biserial | Raw p | Holm p |', '|---|---|---|---|---|---|---|---|---|']
    tex = [r'\begin{tabular}{rlllrrrr}',r'\hline',r'Vehicles & App & Baseline & Metric & $n$ & Difference [95\% CI] & $r_{rb}$ & $p_{Holm}$ \\',r'\hline']
    esc = lambda s: s.replace('_',r'\_')
    for s in stats:
        diff = f"{s['mean_improvement']:.4f} [{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]"
        md.append(f"| {s['vehicles']} | {s['app']} | {s['baseline']} | {s['metric']} | {s['n_pairs']} | {diff} | {s['rank_biserial']:.3f} | {s['p_raw']:.6g} | {s['p_holm']:.6g} |")
        tex.append(f"{s['vehicles']} & {esc(s['app'])} & {esc(s['baseline'])} & {esc(s['metric'])} & {s['n_pairs']} & {diff} & {s['rank_biserial']:.3f} & {s['p_holm']:.6g}" + r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (out/'table.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'table.tex').write_text('% Standalone tabular; adjust layout to journal template.\n'+'\n'.join(tex)+'\n',encoding='utf-8')
    (out/'methods.txt').write_text(__doc__+'\n\n'+note+f'\nArguments: {a}\n',encoding='utf-8')
    print(note)
    print(f'Extracted {len(raw)} logs; calculated {len(stats)} comparisons. Outputs: {out.resolve()}')

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        sys.exit(f'ERROR: {exc}')
