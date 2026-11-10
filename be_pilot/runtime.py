"""Qualification of the user's existing runtime; no installation changes."""
from __future__ import annotations
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from .common import REPO, atomic, read, sha, digest, verify_manifest, read_data

DEFAULT_BINARY='/leonardo/home/userexternal/gbasile0/lammps-glob-dev/build-MOL/lmp'


def command(args):
    r=subprocess.run(args,text=True,capture_output=True)
    if r.returncode: raise RuntimeError(f'{args[0]} failed ({r.returncode}): {r.stderr[-2000:]} {r.stdout[-1000:]}')
    return r.stdout


def input_checks(inputs):
    verify_manifest(REPO); verify_manifest(inputs)
    cfg=read(REPO/'campaign.json')
    if sha(Path(inputs)/'mace.model')!=cfg['model_raw_sha256']: raise ValueError('Wrong trained MACE checkpoint')
    groups=[]; seeds=[]
    for p in cfg['parents']:
        if sha(REPO/p['seed_file'])!=p['seed_sha256']: raise ValueError('Changed parent structure')
        xyz,box=read_data(REPO/p['seed_file'])
        from .common import check_geometry,TYPES,cn
        import numpy as np
        check_geometry(TYPES,xyz,box,np.array(cfg['box_bounds_A']))
        if abs(cn(xyz,box,2.5)-p['seed_cn25'])>2e-8 or abs(cn(xyz,box,2.7)-p['seed_cn27'])>6e-6: raise ValueError('Exported seed CV mismatch')
        groups.append(p['source_group'])
        for role in (['primary','control'] if p['control'] else ['primary']):
            from .common import seed
            for i in range(4):
                seeds.extend([seed(p['parent_id'],f'{role}_rep{i}','velocity'),seed(p['parent_id'],f'{role}_rep{i}','thermostat')])
    if len(groups)!=60 or len(set(groups))!=60: raise ValueError('Expected 60 distinct source groups')
    if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate production random seeds')
    import numpy as np
    for cv in ['R25','R27']:
        folder=Path(inputs)/'baseline'/cv
        fes=np.loadtxt(folder/'fes.dat'); selection=read(folder/'derivative_selection.json')
        curvature=np.gradient(fes[:,2],fes[:,0]);lo,hi=selection['interval_CN']
        ids=np.flatnonzero((fes[:,0]>lo)&(fes[:,0]<hi));k=ids[np.argmax(curvature[ids])]
        if curvature[k]<=0 or abs(fes[k,0]-selection['target_CN'])>1e-12:
            raise ValueError('Curvature target verification failed: '+cv)
        fit=read(folder/'configuration.json');edges=np.array(fit['edges'])
        if not (edges[0]+(edges[1]-edges[0])/2<fes[k,0]<edges[-1]-(edges[-1]-edges[-2])/2):
            raise ValueError('Target outside diffusion support')
    return dict(parents=len(groups),primary_shots=240,control_shots=24,unique_production_seeds=len(seeds),repository_manifest=sha(REPO/'SHA256SUMS.json'),external_manifest=sha(Path(inputs)/'SHA256SUMS.json'))


def ensure_model(inputs, run):
    """Convert exactly once on an allocated GPU, under the active environment."""
    inputs=Path(inputs); run=Path(run); folder=run/'runtime'; folder.mkdir(parents=True,exist_ok=True)
    target=folder/'model.pt'; marker=folder/'model.json'
    import torch
    versions={name:importlib.metadata.version(name) for name in ['torch','mace-torch','e3nn']}
    for name in ['cuequivariance','cuequivariance-torch','cuequivariance-ops-torch-cu12']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    signature=dict(raw_sha256=sha(inputs/'mace.model'),python=sys.executable,versions=versions)
    with (folder/'.model.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.exists():
            old=read(marker)
            if old['signature']!=signature or sha(target)!=old['model_sha256']:
                raise ValueError('Converted model/runtime changed; use a separate run directory')
        elif target.exists(): raise ValueError('Unverified existing converted model: '+str(target))
        else:
            converter=shutil.which('mace_create_lammps_model')
            if not converter: raise RuntimeError('mace_create_lammps_model is unavailable in the active Python environment')
            with tempfile.TemporaryDirectory(prefix='be-mace-',dir=folder) as temp:
                raw=Path(temp)/'mace.model'; shutil.copy2(inputs/'mace.model',raw)
                output=command([converter,str(raw),'--format=mliap'])
                (folder/'conversion.log').write_text(output)
                exported=Path(str(raw)+'-mliap_lammps.pt')
                torch.load(exported,map_location='cpu',weights_only=False)
                shutil.copy2(exported,target)
            atomic(marker,dict(signature=signature,model_sha256=sha(target)))
        torch.load(target,map_location='cpu',weights_only=False)
    return target,versions


def mpi_links(path):
    out=command(['ldd',str(path)])
    if 'not found' in out: raise RuntimeError('Unresolved runtime libraries:\n'+out)
    libs=[]
    for line in out.splitlines():
        if 'libmpi.so' in line and '=>' in line:
            libs.append(str(Path(line.split('=>')[1].strip().split()[0]).resolve()))
    return out,libs


def qualify(inputs, run):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU qualification must run inside a Slurm allocation')
    checks=input_checks(inputs)
    binary=Path(os.environ.get('LAMMPS_BIN',DEFAULT_BINARY)).expanduser().resolve()
    if not os.access(binary,os.X_OK): raise RuntimeError('LAMMPS executable missing: '+str(binary))
    kernel=os.environ.get('PLUMED_KERNEL')
    if not kernel: raise RuntimeError('Set PLUMED_KERNEL in your runtime.env to the working MPI-enabled PLUMED kernel')
    kernel=Path(kernel).expanduser().resolve()
    plumed=os.environ.get('PLUMED_BIN') or shutil.which('plumed')
    if not plumed: raise RuntimeError('Set PLUMED_BIN or add the matching PLUMED executable to PATH')
    command([plumed,'--has-mpi'])
    ldd_binary,mpi_binary=mpi_links(binary); ldd_kernel,mpi_kernel=mpi_links(kernel)
    if not mpi_binary or not mpi_kernel or set(mpi_binary)!=set(mpi_kernel):
        raise RuntimeError('LAMMPS and PLUMED must resolve to the same MPI library')
    cache=binary.parent/'CMakeCache.txt'
    cmake=cache.read_text() if cache.exists() else ''
    mode=next((line.split('=',1)[1] for line in cmake.splitlines() if line.startswith('PLUMED_MODE:')),None)
    if mode!='runtime':
        resolved=[line.split('=>',1)[1].strip().split()[0] for line in ldd_binary.splitlines() if 'plumed' in line.lower() and '=>' in line]
        if not any(Path(p).resolve()==kernel for p in resolved):
            raise RuntimeError('Cannot verify PLUMED linkage: use a runtime-linked build or identify its exact shared kernel')
    helptext=command([str(binary),'-h'])
    for package in ['ML-IAP','KOKKOS','PLUMED']:
        if package not in helptext: raise RuntimeError('LAMMPS package missing: '+package)
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Active PyTorch cannot access CUDA')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if devices==['']:
        devices=command(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).splitlines()
    if len(devices)!=4 or torch.cuda.device_count()!=4: raise RuntimeError('Expected exactly four allocated, visible GPUs')
    model,versions=ensure_model(inputs,run)
    identity=dict(binary_sha256=sha(binary),model_sha256=sha(model),kernel_sha256=sha(kernel),mpi=mpi_binary,
        python=sys.executable,versions=versions,repository_manifest=checks['repository_manifest'],external_manifest=checks['external_manifest'])
    result=dict(**identity,binary=str(binary),model=str(model),kernel=str(kernel),devices=devices,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(4)],fingerprint=digest(identity),box=read(REPO/'campaign.json')['box_bounds_A'],
        checks=checks,plumed_mode=mode,ldd_lammps=ldd_binary,ldd_plumed=ldd_kernel,job_id=os.environ['SLURM_JOB_ID'])
    atomic(Path(run)/'runtime'/f'preflight_{os.environ["SLURM_JOB_ID"]}.json',result)
    return result
