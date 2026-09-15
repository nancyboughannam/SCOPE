#!/usr/bin/env python3
"""Create an IEEE-style catalogue sensitivity table from local results.

INSTALL: python3 -m pip install numpy matplotlib
RUN: python3 catalogue_results_table.py --sim-root /path/to/simulation_results --decision-root /path/to/catalogue_runs
Defaults: iterations 1..10, subset seeds 1 2, 1800 vehicles, 59 decisions/run.
For completed pilots only add --iterations 1 (output marked PILOT).

Simulation folder names supported:
  ite1-50%-seed2/<...ALL_APPS_GENERIC.log>
  pilot_half_s2/sim/ite1/<...ALL_APPS_GENERIC.log>
  half_s2/ite1/<...ALL_APPS_GENERIC.log>
Equivalent names: full/half/quarter or 100%/50%/25%.
Use a distinct iteration folder for each Java run. For older ite1-50% naming,
missing subset seed defaults to 1. Decision identity comes from CSV columns.
Roots may be directories or ZIP files. Directories are searched recursively;
ZIP files inside a directory are not expanded (pass a ZIP as the root instead).

Outputs: catalogue_table.pdf/png/tex/md, run_results.csv, summary.csv,
subset_summary.csv, audit.txt. No significance claims are generated.
Simulation percentage uses the exact supplied MATLAB formula; service is seconds.

Main table: average the two subset values WITHIN each simulation iteration, then
compute mean and sample SD across the 10 iterations. Full catalogue uses one run
per iteration. This does not claim 20 independent runs. subset_summary.csv reports
each subset separately so composition effects remain visible. Timing summarizes
per-run means (all decision calls, including first); no startup/HTTP timing.
Full repeats at different subset seeds are redundant: select the lowest subset
seed, verify performance/churn agree, and record the excluded repeat in audit.txt.
Duplicates within the same identity are rejected. No silent missing-run skipping.

Example (paths are illustrative - point these at your own simulator output
and this repo's FLaskAPIs/catalogue_runs directory):

python3 catalogue_sensitivity/catalogue_results_table.py \
  --sim-root "/path/to/ThreeBrains/sim_results" \
  --decision-root "/path/to/FLaskAPIs/catalogue_runs" \
  --iterations 5 \
  --output "/path/to/FLaskAPIs/catalogue_table_2_iterations"


"""
import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
import numpy as np

METRICS = ['failure_pct','service_s','reconfigs','total_churn','planning_ms']
LABELS = ['Failure (%)','Service (s)','Reconfigs.','Total churn','Planning (ms)']
DECIMALS = [2,4,2,4,1]

def entries(root, suffix):
    root=Path(root).expanduser()
    if root.is_file() and zipfile.is_zipfile(root):
        with zipfile.ZipFile(root) as z:
            for n in sorted(z.namelist()):
                if '__MACOSX' not in n and n.endswith(suffix):
                    yield n,z.read(n).decode('utf-8-sig')
    elif root.is_dir():
        for p in sorted(root.rglob('*'+suffix)):
            if '__MACOSX' not in p.parts:
                yield str(p),p.read_text(encoding='utf-8-sig')
    else: raise ValueError(f'Not a directory or ZIP: {root}')

def sim_identity(name):
    it=re.search(r'(?:^|[/_\-])ite(?:ration)?(\d+)(?=[/_\-.]|$)',name)
    if not it: raise ValueError(f'Cannot find iteration in simulation path: {name}')
    size=re.search(r'(100|50|25)%',name)
    if size: fraction=int(size[1])/100
    else:
        sizes=[f for word,f in [('full',1.),('half',.5),('quarter',.25)] if re.search(r'(?:^|[/_\-])'+word+r'(?=[/_\-]|$)',name)]
        if len(sizes)!=1: raise ValueError(f'Cannot find full/half/quarter or percentage in: {name}')
        fraction=sizes[0]
    seed=re.search(r'(?:seed|_s)(\d+)(?=[/_.\-]|$)',name)
    return int(it[1]),fraction,int(seed[1]) if seed else 1

def csv_out(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def mean_sd(values):
    x=np.asarray(values,dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x)>1 else None

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sim-root',required=True);p.add_argument('--decision-root',required=True)
    p.add_argument('--output',default='catalogue_table_results')
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--subset-seeds',type=int,nargs='+',default=[1,2])
    p.add_argument('--vehicles',type=int,default=1800)
    p.add_argument('--expected-decisions',type=int,default=59)
    a=p.parse_args()
    if a.iterations<1 or len(set(a.subset_seeds))!=len(a.subset_seeds):p.error('Invalid iterations or duplicate subset seeds')
    sims={}; decisions={};audit=[]
    suffix=f'RT_OPT_ML_{a.vehicles}DEVICES_ALL_APPS_GENERIC.log'
    for name,text in entries(a.sim_root,suffix):
        key=sim_identity(name)
        if key[0]>a.iterations or (key[1]!=1 and key[2] not in a.subset_seeds):continue
        if key in sims:raise ValueError(f'Duplicate simulation {key}: {name}')
        x=list(map(float,text.splitlines()[1].split(';')))
        if len(x)<5 or not np.isfinite(x).all() or min(x[0],x[1],x[4])<0 or x[0]<=0:raise ValueError(f'Invalid metric data: {name}')
        sims[key]=dict(failure_pct=100*x[1]/(x[0]+x[1]),service_s=x[4],completed=x[0],failed=x[1],simulation_file=name)
    for name,text in entries(a.decision_root,'.csv'):
        if 'decision_logs' not in name:continue
        rows=list(csv.DictReader(io.StringIO(text)))
        if not rows:raise ValueError(f'Empty decision file: {name}')
        r=rows[0]
        if int(r['num_devices'])!=a.vehicles or r['policy']!='RT_OPT_ML':continue
        key=(int(r['iteration']),float(r['catalogue_fraction']),int(r['catalogue_subset_seed']))
        if key[0]>a.iterations or (key[1]!=1 and key[2] not in a.subset_seeds):continue
        if key in decisions:raise ValueError(f'Duplicate decision log {key}: {name}')
        for field in ['iteration','simulation_seed','num_devices','lambda','w_failure','w_service','r_max','catalogue_fraction','catalogue_subset_seed','catalogue_sha256','full_catalogue_count','selected_catalogue_count','hard_stability_enabled','policy']:
            if len({x[field] for x in rows})!=1:raise ValueError(f'Mixed {field}: {name}')
        if r['hard_stability_enabled'].lower()!='true':raise ValueError(f'Hard bound disabled: {name}')
        ts=np.array([float(x['timestamp']) for x in rows])
        if len(rows)!=a.expected_decisions or len(set(ts))!=len(ts) or not np.allclose(np.diff(ts),60):raise ValueError(f'Incomplete/duplicated decision sequence: {name}')
        costs=[]
        for x in rows:
            c=(abs(float(x['selected_alpha'])-float(x['previous_alpha']))/.9+sum(abs(float(x[f'selected_w_load_{i}'])-float(x[f'previous_w_load_{i}']))/.8+abs(float(x[f'selected_threshold_{i}'])-float(x[f'previous_threshold_{i}']))/75 for i in range(3)))/7
            if not np.isclose(c,float(x['reconfiguration_cost']),atol=1e-10,rtol=0):raise ValueError(f'Churn mismatch: {name}')
            costs.append(c)
        times=np.array([float(x['optimizer_time_ms']) for x in rows])
        if not np.isfinite(times).all() or min(times)<0:raise ValueError(f'Invalid timing: {name}')
        decisions[key]=dict(simulation_seed=int(r['simulation_seed']),lambda_value=float(r['lambda']),r_max=float(r['r_max']),w_failure=float(r['w_failure']),w_service=float(r['w_service']),catalogue_count=int(r['selected_catalogue_count']),full_count=int(r['full_catalogue_count']),catalogue_hash=r['catalogue_sha256'],reconfigs=sum(c>1e-10 for c in costs),total_churn=sum(costs),max_churn=max(costs),violations=sum(c>float(r['r_max'])+1e-10 for c in costs),planning_ms=float(times.mean()),planning_median_ms=float(np.median(times)),planning_max_ms=float(times.max()),decision_file=name)
    required=[(i,f,s) for i in range(1,a.iterations+1) for f in [.25,.5] for s in a.subset_seeds]
    for i in range(1,a.iterations+1):
        keys=sorted(k for k in sims if k[0]==i and k[1]==1 and k in decisions)
        if not keys:raise ValueError(f'Missing full-catalogue simulation/decisions for iteration {i}')
        chosen=keys[0];required.append(chosen)
        for k in keys[1:]:
            for field in ['failure_pct','service_s','completed','failed']:
                if not np.isclose(sims[k][field],sims[chosen][field],rtol=0,atol=1e-10):raise ValueError(f'Full repeats disagree: {k} vs {chosen}')
            for field in ['simulation_seed','lambda_value','r_max','w_failure','w_service','catalogue_hash','reconfigs','total_churn','max_churn','violations']:
                if decisions[k][field]!=decisions[chosen][field]:raise ValueError(f'Full repeat mismatch in {field}: {k}')
            audit.append(f'Excluded redundant full-catalogue repeat {k}; retained {chosen}. Timing not averaged over repeats.')
    missing=[k for k in required if k not in sims or k not in decisions]
    if missing:raise ValueError(f'Missing simulation or decision data for {missing[:20]} (iteration, fraction, subset seed). No partial table created.')
    records=[]
    for k in sorted(required):
        records.append(dict(iteration=k[0],fraction=k[1],subset_seed=k[2],**sims[k],**decisions[k]))
    for field in ['lambda_value','r_max','w_failure','w_service','full_count']:
        if len({r[field] for r in records})!=1:raise ValueError(f'Experiment settings differ: {field}')
    seen_seeds=[]
    for i in range(1,a.iterations+1):
        ss={r['simulation_seed'] for r in records if r['iteration']==i}
        if len(ss)!=1:raise ValueError(f'Unpaired simulation seeds in iteration {i}')
        seen_seeds.extend(ss)
    if len(set(seen_seeds))!=a.iterations:raise ValueError('Simulation seeds repeated across iterations')
    for f in [.25,.5,1.]:
        if len({r['catalogue_count'] for r in records if r['fraction']==f})!=1:raise ValueError(f'Changing catalogue size: {f}')
        for s in {r['subset_seed'] for r in records if r['fraction']==f}:
            if len({r['catalogue_hash'] for r in records if r['fraction']==f and r['subset_seed']==s})!=1:raise ValueError('Catalogue membership changes between iterations')
    summaries=[]; subset_summaries=[]
    for f in [.25,.5,1.]:
        rr=[r for r in records if r['fraction']==f]
        row=dict(fraction=f,candidates=rr[0]['catalogue_count'],simulation_iterations=a.iterations,subset_selections=len(a.subset_seeds) if f<1 else 1)
        for m in METRICS:
            v=[np.mean([r[m] for r in rr if r['iteration']==i]) for i in range(1,a.iterations+1)]
            row[m+'_mean'],row[m+'_sd']=mean_sd(v)
        row['max_churn_observed']=max(r['max_churn'] for r in rr)
        row['budget_violations']=sum(r['violations'] for r in rr)
        summaries.append(row)
        for s in sorted({r['subset_seed'] for r in rr}):
            sub=[r for r in rr if r['subset_seed']==s];q=dict(fraction=f,subset_seed=s,n=len(sub))
            for m in METRICS:q[m+'_mean'],q[m+'_sd']=mean_sd([r[m] for r in sub])
            subset_summaries.append(q)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    csv_out(out/'run_results.csv',records);csv_out(out/'summary.csv',summaries);csv_out(out/'subset_summary.csv',subset_summaries)
    pilot=a.iterations<10
    title=('PILOT: ' if pilot else '')+f'Catalogue-size sensitivity at {a.vehicles:,} vehicles'
    note=(f'Entries are mean ± sample SD across {a.iterations} simulation iterations. '
          f'For reduced catalogues, {len(a.subset_seeds)} subset selections are averaged within each iteration first. '
          'SD therefore describes simulation-seed variability of these averages; subset-specific results are in subset_summary.csv. '
          'Planning time is the mean per-decision time within each run. '
          f'Lambda = {records[0]["lambda_value"]:g}; Rmax = {records[0]["r_max"]:g}. '
          f'Total observed budget violations: {sum(r["violations"] for r in records)}. '
          'This is a descriptive sensitivity summary, not a significance test.')
    if a.iterations==1:note=note.replace('Entries are mean ± sample SD across 1 simulation iterations.','One simulation iteration only: SD is unavailable; displayed entries are means over subset selections.')
    headers=['Catalogue\n(candidates)']+LABELS+['Max.\nchurn']
    cells=[]
    for r in summaries:
        line=[f'{int(r["fraction"]*100)}% ({r["candidates"]:,})']
        for m,dp in zip(METRICS,DECIMALS):
            text=f'{r[m+"_mean"]:.{dp}f}'
            if r[m+'_sd'] is not None:text+=f' ± {r[m+"_sd"]:.{dp}f}'
            line.append(text)
        line.append(f'{r["max_churn_observed"]:.4f}');cells.append(line)
    md=['# '+title,'',note,'','| '+' | '.join(h.replace('\n',' ') for h in headers)+' |','|'+'---|'*len(headers)]
    md+=['| '+' | '.join(r)+' |' for r in cells]
    (out/'catalogue_table.md').write_text('\n'.join(md)+'\n')
    tex=[r'% Requires \usepackage{booktabs}; two-column-wide table.',r'\begin{table*}[t]',r'\centering',r'\caption{'+title.replace('%',r'\%')+'}',r'\label{tab:catalogue_sensitivity}',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Catalogue & Failure (\%) & Service (s) & Reconfigs. & Total churn & Planning (ms) & Max. churn \\',r'\midrule']
    for line in cells:tex.append(' & '.join(x.replace('%',r'\%').replace('±',r'$\pm$') for x in line)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',r'\begin{minipage}{0.98\textwidth}',r'\footnotesize '+note.replace('±',r'$\pm$').replace('_',r'\_'),r'\end{minipage}',r'\end{table*}']
    (out/'catalogue_table.tex').write_text('\n'.join(tex)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off')
    ax.text(.5,.98,title,ha='center',va='top',fontsize=11,transform=ax.transAxes)
    table=ax.table(cellText=cells,colLabels=headers,cellLoc='center',colLoc='center',bbox=[0,.40,1,.43],colWidths=[.17,.15,.16,.12,.15,.15,.10])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (row,col),cell in table.get_celld().items():
        cell.set_facecolor('white');cell.visible_edges='';cell.set_linewidth(.6)
        if row==0:cell.visible_edges='TB';cell.set_text_props(weight='bold')
        if row==len(cells):cell.visible_edges='B'
    ax.text(0,.32,textwrap.fill(note,145),fontsize=8,ha='left',va='top',transform=ax.transAxes)
    fig.savefig(out/'catalogue_table.pdf',bbox_inches='tight');fig.savefig(out/'catalogue_table.png',dpi=300,bbox_inches='tight');plt.close(fig)
    audit.insert(0,f'Validated {len(records)} run pairs. Arguments: {a}')
    audit.append('No paired significance test performed. Do not use across-decision samples as independent run replicates.')
    (out/'audit.txt').write_text('\n'.join(audit)+'\n')
    print('\n'.join(audit));print(f'Wrote table and data to {out.resolve()}')

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,IndexError) as e:sys.exit(f'ERROR: {e}')
