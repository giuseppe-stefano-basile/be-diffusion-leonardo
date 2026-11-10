"""Immutable attempts, geometry-validated completion, and bounded preparations."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .common import REPO, DT, TYPES, atomic, read, sha, digest, seed, cn, read_data, write_data, dump_frames, colvar, check_geometry
STOP=threading.Event()
CHILDREN=set()
MUTEX=threading.Lock()


def stop(signum=None, frame=None):
    STOP.set()
    with MUTEX:
        for p in list(CHILDREN):
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass


def plumed(parent=None):
    text='UNITS LENGTH=A TIME=ps ENERGY=kj/mol\nion: GROUP ATOMS=1\nwaters: GROUP ATOMS=2-374:3\n'
    for name,r0 in [('cn25',2.5),('cn27',2.7)]:
        text+=f'{name}: COORDINATION GROUPA=ion GROUPB=waters SWITCH={{MATHEVAL FUNC=1/(1+exp(4.0*(x-{r0}))) R_0=1.0 D_0=0.0}}\n'
    args='cn25,cn27'
    if parent is not None:
        cv='cn25' if parent['cv']=='R25' else 'cn27'
        text+=f'lower: LOWER_WALLS ARG={cv} AT={parent["lo"]:.17g} KAPPA=20000.0\nupper: UPPER_WALLS ARG={cv} AT={parent["hi"]:.17g} KAPPA=20000.0\n'
        args+=',lower.bias,upper.bias'
    return text+f'PRINT ARG={args} STRIDE=10 FILE=COLVAR FMT=%.16g\nFLUSH STRIDE=1000\n'


def lammps(duration, velocity_seed, thermostat_seed, restart=False):
    init='read_restart start.restart' if restart else 'read_data start.data'
    velocity='' if restart else f'velocity all create 298.0 {velocity_seed} mom yes rot yes dist gaussian\n'
    return f'''units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
{init}
reset_timestep 0
timestep 0.0005
pair_style mliap unified model.pt 0
pair_coeff * * Be O H
{velocity}fix integrator all nve
fix thermostat all langevin 298.0 298.0 0.5 {thermostat_seed}
fix cv all plumed plumedfile plumed.dat outfile plumed.log
fix_modify cv energy yes
fix_modify cv virial yes
thermo_style custom step time temp press vol density etotal epair fmax
thermo 20
thermo_modify flush yes
dump trajectory all custom 10 trajectory.lammpstrj id type x y z
dump_modify trajectory sort id format float %.16g
run {round(duration/DT)}
write_restart final.restart
write_data final.data
'''


def completed(folder, key=None):
    marker=Path(folder)/'COMPLETE.json'
    if not marker.exists(): return None
    m=read(marker)
    if key is not None and m['input_key']!=key: raise ValueError('Completed stage input changed: '+str(folder))
    attempt=Path(folder)/m['attempt']
    for name,expected in m['outputs'].items():
        if sha(attempt/name)!=expected: raise ValueError('Completed output changed: '+str(attempt/name))
    return attempt,m


def validate(attempt, duration, box, restrained=False):
    attempt=Path(attempt)
    text=(attempt/'plumed.dat').read_text()
    if not restrained and any(w in text.upper() for w in ('WALLS','RESTRAINT','METAD')):
        raise ValueError('Biased production input')
    data=colvar(attempt/'COLVAR'); expected=np.arange(round(duration/.005)+1)*.005
    if len(data)!=len(expected) or not np.allclose(data[:,0],expected,rtol=0,atol=1e-8):
        raise ValueError('Incomplete or irregular COLVAR timeline')
    n=0
    for n,(step,xyz,bounds) in enumerate(dump_frames(attempt/'trajectory.lammpstrj'),1):
        j=n-1
        if j>=len(data) or abs(step*DT-data[j,0])>1e-8: raise ValueError('Coordinate/CV times differ')
        check_geometry(TYPES,xyz,bounds,np.asarray(box))
        if max(abs(cn(xyz,bounds,2.5)-data[j,1]),abs(cn(xyz,bounds,2.7)-data[j,2]))>2e-8:
            raise ValueError('Coordinate/CV disagreement')
    if n!=len(data): raise ValueError('Missing coordinate frames')
    xyz,bounds=read_data(attempt/'final.data'); check_geometry(TYPES,xyz,bounds,np.asarray(box))
    if (attempt/'final.restart').stat().st_size==0: raise ValueError('Empty restart')
    log=(attempt/'log.lammps').read_text()
    if 'Loop time of' not in log or 'ERROR' in log: raise ValueError('LAMMPS did not complete cleanly')
    thermo=[]; active=False
    for line in log.splitlines():
        fields=line.split()
        if len(fields)==9 and fields[:3]==['Step','Time','Temp']: active=True; continue
        if active and len(fields)==9:
            try: thermo.append([float(x) for x in fields])
            except ValueError: active=False
    if not thermo or not np.isfinite(thermo).all(): raise ValueError('Missing/nonfinite thermodynamic diagnostics')
    temperatures=np.array(thermo)[:,2]
    return dict(temperature_min_K=float(temperatures.min()),temperature_max_K=float(temperatures.max()),frames=n,duration_ps=duration,cn25_min=float(data[:,1].min()),cn25_max=float(data[:,1].max()),
        cn27_min=float(data[:,2].min()),cn27_max=float(data[:,2].max()),both_CVs_reconstructed=True,production_unbiased=not restrained)


def stage(folder, initial, duration, parent, gpu, runtime, restrained=False, restart=False, label='production'):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    velocity_seed=seed(parent['parent_id'],label,'velocity')
    thermostat_seed=seed(parent['parent_id'],label,'thermostat')
    meta=dict(initial_sha256=sha(initial),model_sha256=runtime['model_sha256'],binary_sha256=runtime['binary_sha256'],
        duration_ps=duration,parent_id=parent['parent_id'],source_group=parent['source_group'],restrained=restrained,restart=restart,
        plumed=plumed(parent if restrained else None),input=lammps(duration,velocity_seed,thermostat_seed,restart),
        velocity_seed=None if restart else velocity_seed,thermostat_seed=thermostat_seed,software_fingerprint=runtime['fingerprint'])
    key=digest(meta)
    old=completed(folder,key)
    if old: return old
    if STOP.is_set(): raise InterruptedError('Stopping before new stage')
    i=0
    while (folder/f'attempt_{i:03d}').exists(): i+=1
    attempt=folder/f'attempt_{i:03d}'; attempt.mkdir()
    atomic(attempt/'input_record.json',meta)
    (attempt/'plumed.dat').write_text(meta['plumed']); (attempt/'in.lammps').write_text(meta['input'])
    (attempt/('start.restart' if restart else 'start.data')).symlink_to(Path(initial).resolve())
    (attempt/'model.pt').symlink_to(Path(runtime['model']).resolve())
    command=[runtime['binary'],'-k','on','g','1','-sf','kk','-pk','kokkos','gpu/aware','on','newton','on','neigh','half','-log','log.lammps','-in','in.lammps']
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['OMP_NUM_THREADS']='1'
    start=time.monotonic()
    try:
        with (attempt/'stdout.log').open('w') as out:
            with MUTEX:
                if STOP.is_set(): raise InterruptedError('Stopping')
                p=subprocess.Popen(command,cwd=attempt,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                CHILDREN.add(p)
            try: code=p.wait()
            finally:
                with MUTEX: CHILDREN.discard(p)
        if code or STOP.is_set(): raise RuntimeError(f'Interrupted/failed LAMMPS process ({code})')
        validation=validate(attempt,duration,runtime['box'],restrained)
        m=dict(input_key=key,attempt=attempt.name,elapsed_seconds=time.monotonic()-start,validation=validation,
            outputs={name:sha(attempt/name) for name in ['COLVAR','trajectory.lammpstrj','final.data','final.restart','in.lammps','plumed.dat','input_record.json','log.lammps']})
        atomic(folder/'COMPLETE.json',m)
        return attempt,m
    except BaseException as e:
        atomic(attempt/'FAILED.json',dict(error=str(e),elapsed_seconds=time.monotonic()-start))
        raise


def prepare(run, parent, gpu, runtime, level=8):
    root=Path(run)/'preparations'/parent['parent_id']/f'hold_{level}'
    marker=root/'SELECTED.json'
    if marker.exists():
        m=read(marker)
        if sha(root/'selected.data')!=m['selected_sha256']: raise ValueError('Selected preparation changed')
        prior=completed(root/m['stage'])
        if prior is None: raise ValueError('Selected preparation lost completion record')
        return root/'selected.data',m
    if (root/'QUARANTINE.json').exists(): return None,None
    cfg=read(REPO/'campaign.json'); elapsed=0.; restart=False
    initial=REPO/parent['seed_file']
    if level==16:
        _,base=prepare(run,parent,gpu,runtime,8)
        if base is None: return None,None
        elapsed=base['actual_preparation_ps']; initial=Path(base['final_restart']); restart=True
    root.mkdir(parents=True,exist_ok=True)
    # First reach the prescribed preparation length; extensions are in 2 ps pieces.
    while elapsed < level+4-1e-8:
        duration=level-elapsed if elapsed<level else 2.
        name=f'segment_{elapsed:g}_{elapsed+duration:g}'
        attempt,_=stage(root/name,initial,duration,parent,gpu,runtime,True,restart,label=f'prep{level}_{elapsed:g}')
        elapsed+=duration
        candidates=[]
        for step,xyz,box in dump_frames(attempt/'trajectory.lammpstrj'):
            value=cn(xyz,box,2.5 if parent['cv']=='R25' else 2.7)
            if step*DT>=duration-2.-1e-8 and parent['lo']<=value<parent['hi']:
                candidates.append((step,xyz,box,value))
        if candidates:
            choice=np.random.default_rng(seed(parent['parent_id'],level,'tail')).integers(len(candidates))
            step,xyz,box,value=candidates[int(choice)]
            write_data(root/'selected.data',xyz,box)
            m=dict(parent_id=parent['parent_id'],source_group=parent['source_group'],requested_preparation_ps=level,
                actual_preparation_ps=elapsed,selected_step=int(step),selected_CN=value,stage=name,
                selected_sha256=sha(root/'selected.data'),final_restart=str((attempt/'final.restart').resolve()),
                source_stage_input_key=read(root/name/'COMPLETE.json')['input_key'])
            atomic(marker,m); return root/'selected.data',m
        initial=attempt/'final.restart'; restart=True
    atomic(root/'QUARANTINE.json',dict(reason='No eligible frame in final 2 ps after bounded preparation',actual_preparation_ps=elapsed,parent_id=parent['parent_id']))
    return None,None


def run_parent(run, parent, gpu, runtime, smoke=False):
    cfg=read(REPO/'campaign.json')
    if smoke:
        pre,_=stage(Path(run)/'smoke'/parent['parent_id']/'restrained',REPO/parent['seed_file'],.05,parent,gpu,runtime,restrained=True,label='smoke_prep')
        stage(Path(run)/'smoke'/parent['parent_id']/'unbiased',pre/'final.data',.1,parent,gpu,runtime,label='smoke_unbiased')
        return
    for level in ([8,16] if parent['control'] else [8]):
        initial,meta=prepare(run,parent,gpu,runtime,level)
        if initial is None: continue
        for rep in range(4):
            role='primary' if level==8 else 'control'
            folder=Path(run)/'production'/role/parent['parent_id']/f'rep_{rep:02d}'
            attempt,m=stage(folder,initial,2.,parent,gpu,runtime,label=f'{role}_rep{rep}')
            atomic(folder/'record.json',dict(parent_id=parent['parent_id'],source_group=parent['source_group'],role=role,
                rep=rep,requested_preparation_ps=level,actual_preparation_ps=meta['actual_preparation_ps'],
                COLVAR=str((attempt/'COLVAR').relative_to(Path(run))),input_key=m['input_key']))


def execute(run, runtime, smoke=False):
    import queue
    run=Path(run); run.mkdir(parents=True,exist_ok=True)
    lock=(run/'.execution.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another campaign job already holds this run directory')
    STOP.clear()
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]: signal.signal(sig,stop)
    cfg=read(REPO/'campaign.json'); parents=cfg['parents'][:4] if smoke else cfg['parents']
    devices=runtime['devices']; q=queue.Queue()
    for p in parents: q.put(p)
    errors=[]
    def worker(gpu):
        while not STOP.is_set():
            try: parent=q.get_nowait()
            except queue.Empty: return
            try: run_parent(run,parent,gpu,runtime,smoke)
            except Exception as e:
                with MUTEX: errors.append(dict(parent=parent['parent_id'],error=str(e)))
                stop(); return
            finally: q.task_done()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker,devices))
    atomic(run/('smoke_status.json' if smoke else 'execution_status.json'),dict(errors=errors,interrupted=STOP.is_set()))
    if errors or STOP.is_set(): raise RuntimeError('Run interrupted or failed; completed stages preserved. Use resume after resolving the error.')
    if smoke:
        atomic(run/'QUALIFIED.json',dict(fingerprint=runtime['fingerprint'],campaign_sha256=sha(REPO/'campaign.json'),
            repository_manifest_sha256=sha(REPO/'SHA256SUMS.json'),stages=[read(run/'smoke'/p['parent_id']/'unbiased'/'COMPLETE.json') for p in parents]))
