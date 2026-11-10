"""Portable CPU audit, unchanged-prior diffusion refit, and native kinetics."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from .common import REPO,atomic,read,sha,digest,verify_manifest,seed,colvar
sys.path.insert(0,str(REPO/'vendor'))
from counting import transition_counts_hummer
from .kinetics import Passage,diagnostics,baseline_regression
VARIANTS=['primary','lag-double','discard-0p2','prep8-matched','prep16','bins32','bins48']+[f'heldout-{i}' for i in range(5)]


def npz_atomic(path,**values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        np.savez_compressed(f,**values); temp=f.name
    os.replace(temp,path)


def write_csv(path,rows):
    rows=list(rows)
    if not rows: return
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def series(baseline,cv):
    with np.load(Path(baseline)/f'series_{cv}.npz') as z: data=z['data'].copy(); offsets=z['offsets'].copy()
    meta=read(Path(baseline)/'series_provenance.json')
    return [(data[a:b],r['source_group']) for a,b,r in zip(offsets[:-1],offsets[1:],meta)]


def settings(baseline,cv,variant):
    cfg=read(Path(baseline)/cv/'configuration.json'); cfg=dict(cfg)
    cfg['frame_dt_ps']=.1 if cv=='R25' else .005
    if variant=='lag-double': cfg['lag_ps']*=2
    if variant=='discard-0p2': cfg['discard_ps']=.2
    if variant in ['bins32','bins48']:
        n=int(variant[4:]); cfg['edges']=np.linspace(cfg['edges'][0],cfg['edges'][-1],n+1).tolist()
        cfg['bins']=n; cfg['anchor_bin']=int(np.argmin(abs((np.array(cfg['edges'][:-1])+np.array(cfg['edges'][1:]))/2-cfg['anchor_s'])))
    return cfg


def count_one(data,cfg):
    edges=np.array(cfg['edges']); centers=(edges[:-1]+edges[1:])/2
    frame=cfg['frame_dt_ps']; tau=cfg['lag_ps']; lag=round(tau/frame)
    if lag<1 or abs(lag*frame-tau)>1e-10: raise ValueError('Lag is unsupported by saved baseline spacing')
    raw_dt=float(np.median(np.diff(data[:,0]))) if len(data)>1 else frame
    stride=round(frame/raw_dt)
    if stride<1 or abs(stride*raw_dt-frame)>1e-7: raise ValueError('Saved frame spacing incompatible with frozen convention')
    a=data[::stride]
    if len(a)>1 and not np.allclose(np.diff(a[:,0]),frame,rtol=0,atol=1e-7): raise ValueError('Irregular time series')
    a=a[a[:,0]>=cfg['discard_ps']-1e-8]
    counts,valid,far=transition_counts_hummer(a[:,1],edges,centers,lag,lag,True,False)
    return counts,valid,far


def production_rows(run):
    from .engine import completed
    rows=[]
    for p in sorted((Path(run)/'production').glob('*/*/rep_*')):
        done=completed(p)
        if done is None: continue
        attempt,m=done
        # Reconstruct metadata from hashed stage inputs; don't trust mutable indexes.
        record=read(attempt/'input_record.json')
        if record['restrained'] or record['duration_ps']!=2.: raise ValueError('Non-production stage in primary tree')
        role=p.parents[1].name
        if role not in ['primary','control']: raise ValueError('Unknown production role')
        data=colvar(attempt/'COLVAR')
        rows.append(dict(data=data,source_group=record['source_group'],parent_id=record['parent_id'],role=role,
            path=str(attempt/'COLVAR'),sha256=sha(attempt/'COLVAR'),input_key=m['input_key']))
    keys=[r['input_key'] for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate production records')
    return rows


def count_dataset(baseline,cv,rows,variant):
    cfg=settings(baseline,cv,variant); groups={}; newgroups={}; heldout={}; base_total=np.zeros((cfg['bins'],cfg['bins']),dtype=np.int64)
    def add(mapping,g,c):
        if g not in mapping: mapping[g]=np.zeros_like(base_total)
        mapping[g]+=c
    fold=int(variant[-1]) if variant.startswith('heldout-') else None
    def is_held(g): return fold is not None and int(digest(g)[:8],16)%5==fold
    for data,g in series(baseline,cv):
        c,_,_=count_one(data,cfg); base_total+=c
        add(heldout if is_held(g) else groups,g,c)
    if variant=='primary':
        expected=np.load(Path(baseline)/cv/'transition_counts.npy')
        if not np.array_equal(base_total,expected): raise ValueError('Frozen baseline counts not reproduced: '+cv)
    control_ids={p['parent_id'] for p in read(REPO/'campaign.json')['parents'] if p['control']}
    for r in rows:
        if (r['role']=='control') != (variant=='prep16'): continue
        if variant=='prep8-matched' and r['parent_id'] not in control_ids: continue
        a=r['data'][:,[0,1 if cv=='R25' else 2]]
        c,_,_=count_one(a,cfg);g=r['source_group']
        add(newgroups,g,c)
        add(heldout if is_held(g) else groups,g,c)
    total=sum(groups.values(),np.zeros_like(base_total))
    return cfg,total,groups,newgroups,heldout


def audit(inputs,run,out):
    from .runtime import input_checks
    input_checks(inputs)
    baseline=Path(inputs)/'baseline'; out=Path(out);out.mkdir(parents=True,exist_ok=True)
    regression=baseline_regression(baseline); rows=production_rows(run);coverage=[]
    for cv in ['R25','R27']:
        for variant in VARIANTS:
            cfg,counts,groups,newgroups,heldout=count_dataset(baseline,cv,rows,variant)
            folder=out/'counts'/cv/variant;folder.mkdir(parents=True,exist_ok=True)
            npz_atomic(folder/'counts.npz',counts=counts,groups=np.stack(list(groups.values())),group_ids=np.array(list(groups)),
                heldout=sum(heldout.values(),np.zeros_like(counts)))
            atomic(folder/'configuration.json',cfg)
            n=sum(newgroups.values(),np.zeros_like(counts))
            npz_atomic(folder/'new_group_counts.npz',counts=n,group_ids=np.array(list(newgroups)),groups=np.stack(list(newgroups.values())) if newgroups else np.empty((0,*counts.shape)))
            if variant=='primary':
                for i,edge in enumerate(cfg['edges'][1:-1]):
                    coverage.append(dict(CV=cv,interface_CN=edge,new_forward=int(n[i+1,i]),new_reverse=int(n[i,i+1]),
                        baseline_forward=int(counts[i+1,i]-n[i+1,i]),baseline_reverse=int(counts[i,i+1]-n[i,i+1]),
                        new_source_groups=sum(bool(c[:,i].sum() or c[:,i+1].sum()) for c in newgroups.values())))
    write_csv(out/'coverage.csv',coverage)
    cfg_campaign=read(REPO/'campaign.json')
    base_failed={p.parent.parent.name for p in Path(run).glob('preparations/*/hold_8/QUARANTINE.json')}
    control_failed={p.parent.parent.name for p in Path(run).glob('preparations/*/hold_16/QUARANTINE.json')}
    expected_primary=4*sum(p['parent_id'] not in base_failed for p in cfg_campaign['parents'])
    expected_controls=4*sum(p['control'] and p['parent_id'] not in base_failed|control_failed for p in cfg_campaign['parents'])
    atomic(out/'audit.json',dict(expected_usable_primary=expected_primary,expected_usable_controls=expected_controls,preparations=[read(p) for p in sorted(Path(run).glob('preparations/*/hold_*/SELECTED.json'))],primary_shots=sum(r['role']=='primary' for r in rows),control_shots=sum(r['role']=='control' for r in rows),
        expected_primary=240,expected_controls=24,quarantined=[str(p.relative_to(run)) for p in Path(run).glob('preparations/*/hold_*/QUARANTINE.json')],
        failed_attempts=len(list(Path(run).rglob('FAILED.json'))),baseline_regression=regression,
        inputs_sha256=sha(Path(inputs)/'SHA256SUMS.json'),repository_manifest=sha(REPO/'SHA256SUMS.json'),
        production=[{k:v for k,v in r.items() if k!='data'} for r in rows],
        interpretation='Non-overlapping lag-spaced transitions. All descendants share original source-group ID; no physical independence claim.'))
    plot_coverage(out,coverage)
    return rows


def sample_chain(task):
    import hummer_bounded_bayesian as B
    cv,variant,chain,folder,baseline,steps,burn,thin=task
    folder=Path(folder); cfg=read(folder/'configuration.json')
    with np.load(folder/'counts.npz') as z: counts=z['counts']
    signature=dict(counts_sha256=sha(folder/'counts.npz'),configuration_sha256=sha(folder/'configuration.json'),
        sampler_sha256=sha(REPO/'vendor/hummer_bounded_bayesian.py'),arithmetic_sha256=sha(REPO/'vendor/hummer_high_precision.cpp'),steps=steps,burn=burn,thin=thin)
    directory=folder/f'chain_{chain:02d}';directory.mkdir(exist_ok=True)
    marker=directory/'schedule.json'
    if marker.exists() and read(marker)!=signature: raise ValueError('MCMC schedule/input changed; use a new analysis directory')
    atomic(marker,signature)
    final=folder/f'chain_{chain:02d}.npz'; final_marker=directory/'COMPLETE.json'
    if final_marker.exists():
        if read(final_marker)['signature']!=signature or sha(final)!=read(final_marker)['sha256']: raise ValueError('Completed MCMC chain changed')
        return cv,chain
    n=len(counts);anchor=cfg['anchor_bin']; centers=(np.array(cfg['edges'][:-1])+np.array(cfg['edges'][1:]))/2
    if variant not in ['bins32','bins48']:
        with np.load(Path(baseline)/cv/f'chain_{chain:02d}.npz') as z: g=z['initial_g'].copy(); k=z['initial_k'].copy()
    else:
        pop=counts.sum(axis=0)+counts.sum(axis=1)+1.; full=-np.log(pop/pop[anchor]);g=full[np.arange(n)!=anchor]
        k=np.maximum(counts[np.arange(1,n),np.arange(n-1)],1.)/np.maximum(counts.sum(axis=0)[:-1],1.)/cfg['lag_ps']
    rng=np.random.default_rng(seed(cv,variant,chain,'initial'))
    g=np.clip(g+rng.normal(0,.05,n-1),-9.9,9.9);k=np.clip(k*np.exp(rng.normal(0,.05,n-1)),1e-7,999.)
    if not np.isfinite(B.log_target(g,k,counts,cfg['lag_ps'],10.,1000.,anchor)): raise ValueError('Invalid initial state')
    finished=0; index=0; scales=None; chunks=[]
    while finished<steps:
        length=min((max(100000,burn+thin) if index==0 else 50000),steps-finished)
        chunk=directory/f'chunk_{index:03d}.npz'
        chunk_marker=chunk.with_suffix('.json')
        if chunk.exists() and chunk_marker.exists():
            if read(chunk_marker)['sha256']!=sha(chunk): raise ValueError('MCMC checkpoint changed')
            with np.load(chunk) as z: values={k:z[k].copy() for k in z.files}
            if int(values['executed_steps'])!=length: raise ValueError('MCMC chunk schedule changed')
        else:
            if chunk.exists(): chunk.rename(chunk.with_name(chunk.stem+'_orphan_'+str(os.getpid())+'.npz'))
            values=B.sample(counts,centers,cfg['lag_ps'],g,k,10.,1000.,anchor,steps=length,burn=burn if index==0 else 0,
                thin=thin,seed=seed(cv,variant,chain,index,'MCMC'),proposal_scales=scales)
            values['executed_steps']=length
            npz_atomic(chunk,**values)
            atomic(chunk_marker,dict(sha256=sha(chunk),executed_steps=length))
        chunks.append(values);g=values['g'][-1];k=values['k'][-1];scales=values['final_proposal_scales']
        finished+=length;index+=1
    result={k:np.concatenate([c[k] for c in chunks]) for k in ['D','F','g','k','log_likelihood']}
    npz_atomic(final,**result)
    atomic(final_marker,dict(signature=signature,sha256=sha(final)))
    return cv,chain


def fit(inputs,out,variant,steps,burn,thin,workers):
    if variant not in VARIANTS: raise ValueError('Unknown variant: '+variant)
    if steps<=burn or burn<0 or thin<1: raise ValueError('Invalid MCMC schedule')
    tasks=[(cv,variant,i,str(Path(out)/'counts'/cv/variant),str(Path(inputs)/'baseline'),steps,burn,thin) for cv in ['R25','R27'] for i in range(4)]
    # Compile both checked precision paths before expensive fitting starts.
    import hummer_high_precision as P
    P.library('quad');P.library('mpfr')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(sample_chain,tasks): print('Fitted',result,flush=True)


def kinetics(inputs,out,variant):
    summary=[]; sensitivities=[]; heldout_predictions=[]; baseline=Path(inputs)/'baseline'; figures={}; parameter_diagnostics={}; failures=[]
    for cv in ['R25','R27']:
        folder=Path(out)/'counts'/cv/variant;cfg=read(folder/'configuration.json');x=np.array(cfg['edges'][1:-1])
        chains=[]
        for i in range(4):
            with np.load(folder/f'chain_{i:02d}.npz') as z: chains.append({k:z[k].copy() for k in ['D','F','g','k']})
        parameter_diagnostics[cv]={k:diagnostics(np.stack([c[k] for c in chains])) for k in ['D','g','k']}
        draws=np.stack([c['D'] for c in chains]);median=np.median(draws.reshape(-1,len(x)),axis=0)
        fes=np.loadtxt(baseline/cv/'fes.dat');event=read(baseline/cv/'events.json')
        for profile in ['raw_median','smooth_median_h0p10']:
            passage=Passage(x,fes,event,profile); point=passage.evaluate(median,validate_native=True,exact=True)
            fine=Passage(x,fes,event,profile,n=8001).evaluate(median,exact=True)
            for mode in point:
                err=abs(point[mode]['MFPT_seconds']/fine[mode]['MFPT_seconds']-1)
                if err>.001: raise ValueError('Grid convergence >0.1%: '+cv+'/'+profile+'/'+mode)
                sensitivities.append(dict(CV=cv,profile=profile,variant='grid_8001',mode=mode,relative_rate=fine[mode]['rate_s_inverse']/point[mode]['rate_s_inverse']))
            rates=np.full((*draws.shape[:2],2),np.nan)
            for i,chain in enumerate(draws):
                cache=folder/f'{profile}_rates_chain_{i:02d}.npz'
                signature=digest(dict(chain=sha(folder/f'chain_{i:02d}.npz'),profile=profile,native=sha(REPO/'vendor/native_fpe.py'),code=sha(Path(__file__).with_name('kinetics.py'))))
                marker=cache.with_suffix('.json')
                if cache.exists() and marker.exists() and read(marker)['signature']==signature:
                    with np.load(cache) as z: rates[i]=z['rates']
                    failures.extend(read(marker)['failures']); continue
                failed=[]
                for j,D in enumerate(chain):
                    try:
                        r=passage.evaluate(D,validate_native=j in [0,len(chain)//2,len(chain)-1])
                        rates[i,j]=[r[m]['rate_s_inverse'] for m in ['association','dissociation']]
                    except Exception as e:
                        failed.append(dict(CV=cv,profile=profile,chain=i,draw=j,error=str(e)))
                npz_atomic(cache,rates=rates[i]);atomic(marker,dict(signature=signature,failures=failed));failures.extend(failed)
            for j,mode in enumerate(['association','dissociation']):
                a=rates[:,:,j]; valid=np.isfinite(a)
                if not valid.any(): raise ValueError('All posterior MFPT solves failed')
                q=np.quantile(a[valid],[.025,.5,.975]).tolist()
                diag=diagnostics(a) if valid.all() else dict(passed=[False],reason='Failed draws: chain diagnostics not computed after removing draws')
                old=next(r for r in read(baseline/'summary.json')['results'] if r['CV']==cv and r['profile']==profile and r['mode']==mode)
                width_ratio=(q[2]-q[0])/(old['quantiles']['q975']-old['quantiles']['q025'])
                batches=[b for chain in a for b in np.array_split(chain,20) if len(b)>0 and np.isfinite(b).all()]
                batch_q=np.array([np.quantile(b,[.025,.5,.975]) for b in batches])
                mcse=(batch_q.std(axis=0,ddof=1)/np.sqrt(len(batch_q))).tolist() if len(batch_q)>1 else None
                summary.append(dict(quantile_MCSE=mcse,CV=cv,profile=profile,mode=mode,point=point[mode],q025=q[0],median=q[1],q975=q[2],
                    conditional_width_ratio_to_baseline=width_ratio,attempted=int(a.size),failed=int((~valid).sum()),diagnostics=diag,baseline=old))
            if profile=='raw_median':
                for factor in [.5,2.]:
                    result=Passage(x,fes,event,profile,factor=factor).evaluate(median,exact=True)
                    for m in result: sensitivities.append(dict(CV=cv,profile=profile,variant=f'core_{factor}',mode=m,relative_rate=result[m]['rate_s_inverse']/point[m]['rate_s_inverse']))
                for m,values in [('association',[2.4,2.5] if cv=='R25' else [2.6,2.7]),('dissociation',[5.3,5.4])]:
                    for b in values:
                        r=Passage(x,fes,event,profile,boundaries={m:b}).evaluate(median,exact=True)[m]
                        sensitivities.append(dict(CV=cv,profile=profile,variant=f'reflection_{b}',mode=m,relative_rate=r['rate_s_inverse']/point[m]['rate_s_inverse']))
        if variant.startswith('heldout-'):
            import hummer_high_precision as HP
            import hummer_bounded_bayesian as B
            from scipy.special import logsumexp
            with np.load(folder/'counts.npz') as z: validation_counts=z['heldout']
            gs=np.concatenate([c['g'] for c in chains]);ks=np.concatenate([c['k'] for c in chains])
            ids=np.linspace(0,len(gs)-1,min(200,len(gs)),dtype=int)
            ll=[HP.evaluate(B.relative_to_last(gs[i],cfg['anchor_bin'],len(validation_counts)),ks[i],validation_counts,cfg['lag_ps'],precision='quad') for i in ids]
            heldout_predictions.append(dict(CV=cv,heldout_transition_count=int(validation_counts.sum()),complete_profile_draws=len(ids),
                log_posterior_predictive=float(logsumexp(ll)-np.log(len(ll))),per_transition_log_predictive=float((logsumexp(ll)-np.log(len(ll)))/max(1,validation_counts.sum()))))
        figures[cv]=dict(x=x,draws=draws,fes=fes,event=event)
    destination=Path(out)/'results'/variant;destination.mkdir(parents=True,exist_ok=True)
    atomic(destination/'summary.json',dict(results=summary,global_parameter_diagnostics=parameter_diagnostics,failed_draws=failures,heldout_predictions=heldout_predictions,
        uncertainty='Retained-chain spread conditional on the fixed FES, CV, preparation protocol, model and finite-support priors; not validated uncertainty for completed water exchange.',
        interpretation='Reciprocal MFPT/4: provisional threshold-passage rate per water.',automatic_campaign_expansion=False))
    write_csv(destination/'sensitivities.csv',sensitivities)
    plot_results(destination,summary,figures,baseline)
    lines=['# Be diffusion pilot results','',f'Analysis variant: {variant}.', '',
        '| CV | Profile | Event | Rate/water (s⁻¹) | Conditional 2.5–97.5% spread | Width / baseline | Rate MCMC screen |',
        '|---|---|---|---:|---|---:|---|']
    for r in summary: lines.append(f'| {r["CV"]} | {r["profile"]} | {r["mode"]} | {r["point"]["rate_s_inverse"]:.6g} | {r["q025"]:.6g}–{r["q975"]:.6g} | {r["conditional_width_ratio_to_baseline"]:.3f} | {r["diagnostics"]["passed"]} |')
    all_parameters=all(all(d['passed']) for cv in parameter_diagnostics.values() for d in cv.values())
    all_rates=all(all(r['diagnostics']['passed']) for r in summary)
    lines+=['',f'Global driving-parameter screen passed: {all_parameters}. Rate-scalar screens passed: {all_rates}.',
        '', 'These are provisional threshold-passage estimates. FES, preparation and model uncertainty remain separate; source-block independence is not established.', '',
        'Recommendation: review coverage.csv, preparation controls, lag/bin/discard variants and held-out source-family predictions before expanding sampling. '
        'A narrower retained-chain spread alone does not establish improved physical accuracy. No additional MD is launched by analysis.']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def plot_coverage(out,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cv in zip(axes,['R25','R27']):
        r=[x for x in rows if x['CV']==cv];x=[v['interface_CN'] for v in r]
        for key,label in [('new_forward','New forward'),('new_reverse','New reverse')]: ax.plot(x,[v[key] for v in r],'.-',label=label)
        ax.set(title=cv,xlabel='CN interface',ylabel='Lag-spaced transitions',yscale='symlog');ax.legend()
    fig.tight_layout();fig.savefig(Path(out)/'coverage.pdf');plt.close(fig)


def plot_results(out,summary,figures,baseline):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode in ['association','dissociation']:
        fig,ax=plt.subplots(figsize=(9,5))
        for i,r in enumerate([x for x in summary if x['mode']==mode]):
            x=i; old=r['baseline'];q=old['quantiles']
            ax.vlines(x-.12,q['q025'],q['q975'],color='gray',lw=3);ax.plot(x-.12,old['point_rate_s_inverse'],'o',color='gray')
            ax.vlines(x+.12,r['q025'],r['q975'],color='tab:blue',lw=3);ax.plot(x+.12,r['point']['rate_s_inverse'],'o',color='tab:blue')
        selected=[x for x in summary if x['mode']==mode]
        ax.set_xticks(range(len(selected)),[r['CV']+'\n'+r['profile'] for r in selected],fontsize=8)
        ax.set(yscale='log',ylabel='Provisional rate / water [s⁻¹]',title=mode.capitalize()+' — baseline (gray) / pilot refit (blue)')
        fig.text(.5,.01,'Bars: conditional retained-chain 2.5–97.5% spread; FES/preparation/model uncertainty excluded.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1))
        for ext in ['pdf','png','svg']: fig.savefig(Path(out)/f'{mode}_comparison.{ext}')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for col,cv in enumerate(['R25','R27']):
        data=figures[cv];x=data['x'];q=np.quantile(data['draws'].reshape(-1,len(x)),[.025,.5,.975],axis=0)
        axes[0,col].plot(x,q[1],label='Pilot refit');axes[0,col].fill_between(x,q[0],q[2],alpha=.2)
        with (Path(baseline)/cv/'D_all_bins.csv').open() as f: old=list(csv.DictReader(f))
        axes[0,col].plot([float(r['s']) for r in old],[float(r['median']) for r in old],color='gray',label='Baseline')
        axes[0,col].set(title=cv,yscale='log',ylabel='D [CN²/ps]',xlabel='CN');axes[0,col].legend()
        fes=data['fes'];axes[1,col].plot(fes[:,0],fes[:,1])
        event=data['event']
        for key,color in [('source','green'),('onset','red'),('dissociation','orange')]: axes[1,col].axvline(event[key],color=color,label=key)
        axes[1,col].set(xlim=(2.5,5.3),xlabel='CN',ylabel='F [kJ/mol]');axes[1,col].legend()
    fig.tight_layout();fig.savefig(Path(out)/'diffusion_and_targets.pdf');plt.close(fig)


def _analyze(args):
    from .runtime import input_checks
    input_checks(args.inputs)
    out=(args.analysis_dir or args.run_dir/'analysis').resolve()
    out.mkdir(parents=True,exist_ok=True)
    import importlib.metadata,platform
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','numba']}
    signature=dict(repository_manifest=sha(REPO/'SHA256SUMS.json'),external_manifest=sha(args.inputs/'SHA256SUMS.json'),python=platform.python_version(),packages=versions,loaded_modules=os.environ.get('LOADEDMODULES',''))
    marker=out/'analysis_inputs.json'
    if marker.exists() and read(marker)!=signature: raise ValueError('Analysis inputs changed; use a new analysis directory')
    atomic(marker,signature)
    if args.phase in ['audit','all'] or not (out/'audit.json').exists(): audit(args.inputs,args.run_dir,out)
    variants=VARIANTS if args.variant=='suite' else [args.variant]
    if any(v not in VARIANTS for v in variants): raise ValueError('Unknown analysis variant')
    if args.phase in ['fit','all']:
        audit_info=read(out/'audit.json')
        if audit_info['primary_shots']!=audit_info['expected_usable_primary'] or audit_info['control_shots']!=audit_info['expected_usable_controls']:
            raise ValueError('Primary campaign incomplete; run audit/status and resume before refitting')
        for variant in variants: fit(args.inputs,out,variant,args.steps,args.burn,args.thin,args.workers)
    if args.phase in ['kinetics','all']:
        for variant in variants: kinetics(args.inputs,out,variant)


def analyze(args):
    import fcntl
    out=(args.analysis_dir or args.run_dir/'analysis').resolve()
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.analysis.lock').open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another analysis job already holds this output directory')
        _analyze(args)
