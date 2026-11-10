from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
from .common import REPO, read, atomic, sha


def main():
    parser=argparse.ArgumentParser(description='Portable Be diffusion pilot: no local GPU work unless explicitly invoked in an allocation')
    parser.add_argument('command',choices=['preflight','smoke','submit','status','resume','analyze','_execute'])
    parser.add_argument('--inputs',type=Path,default=os.environ.get('BE_INPUTS'))
    parser.add_argument('--run-dir',type=Path,default=os.environ.get('BE_RUN_DIR'))
    parser.add_argument('--runtime-env',type=Path,default=os.environ.get('BE_RUNTIME_ENV'))
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--smoke-job',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--phase',choices=['audit','fit','kinetics','all'],default='audit')
    parser.add_argument('--analysis-dir',type=Path)
    parser.add_argument('--variant',default='primary')
    parser.add_argument('--steps',type=int,default=480000)
    parser.add_argument('--burn',type=int,default=60000)
    parser.add_argument('--thin',type=int,default=20)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.run_dir is None:
        if not os.environ.get('WORK'): parser.error('Set --run-dir or BE_RUN_DIR (or select a project with WORK defined)')
        args.run_dir=Path(os.environ['WORK'])/'be-diffusion/pilot-v1'
    args.run_dir=args.run_dir.expanduser().resolve()
    if args.inputs is None: parser.error('Set --inputs or BE_INPUTS to the extracted external input directory')
    args.inputs=args.inputs.expanduser().resolve()
    if args.command=='preflight':
        from .runtime import input_checks,qualify
        result=input_checks(args.inputs) if args.cpu_only else qualify(args.inputs,args.run_dir)
        print(__import__('json').dumps(result,indent=2))
    elif args.command in ('smoke','submit','resume'):
        if args.runtime_env is None or not args.runtime_env.expanduser().is_file(): parser.error('Set --runtime-env to your working environment activation script')
        runtime_env=args.runtime_env.expanduser().resolve()
        from .runtime import input_checks
        input_checks(args.inputs)
        args.run_dir.mkdir(parents=True,exist_ok=True)
        smoke=args.command=='smoke'
        if not smoke and not (args.run_dir/'QUALIFIED.json').exists(): raise RuntimeError('Run smoke and wait for QUALIFIED.json before production submission')
        frozen=dict(campaign_sha256=sha(REPO/'campaign.json'),repository_manifest=sha(REPO/'SHA256SUMS.json'),
            external_manifest=sha(args.inputs/'SHA256SUMS.json'),runtime_env_sha256=sha(runtime_env),inputs=str(args.inputs),run_dir=str(args.run_dir))
        manifest=args.run_dir/'handoff.json'
        if manifest.exists() and read(manifest)!=frozen: raise ValueError('Run configuration changed: select a new run directory')
        atomic(manifest,frozen)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        script=args.run_dir/f'job_{args.command}_{stamp}.sh'
        command=['python','-m','be_pilot','_execute','--inputs',str(args.inputs),'--run-dir',str(args.run_dir)]
        if smoke: command+=['--smoke-job']
        lines=['#!/usr/bin/env bash','set -euo pipefail',f'cd {shlex.quote(str(REPO))}',f'source {shlex.quote(str(runtime_env))}',
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
            'export HUMMER_CACHE_DIR='+shlex.quote(str(args.run_dir/'runtime/hummer-cache')),
            'exec '+shlex.join(command)]
        script.write_text('\n'.join(lines)+'\n')
        cmd=['sbatch','--parsable','--account=INF26_biophys','--partition=boost_usr_prod','--nodes=1','--ntasks=1','--cpus-per-task=32','--gres=gpu:4',
            '--job-name=be_diff_'+('smoke' if smoke else 'pilot'),'--signal=B:USR1@120','--output='+str(args.run_dir/'slurm-%j.out'),
            '--error='+str(args.run_dir/'slurm-%j.err')]
        if smoke: cmd+=['--qos=boost_qos_dbg','--time=00:10:00']
        else: cmd+=['--time=24:00:00']
        result=subprocess.run(cmd+[str(script)],check=True,text=True,capture_output=True)
        print(result.stdout.strip())
    elif args.command=='_execute':
        from .runtime import qualify
        from .engine import execute
        runtime=qualify(args.inputs,args.run_dir)
        if not args.smoke_job:
            q=read(args.run_dir/'QUALIFIED.json')
            if q['fingerprint']!=runtime['fingerprint']: raise ValueError('Runtime/input fingerprint differs from smoke qualification')
        execute(args.run_dir,runtime,args.smoke_job)
    elif args.command=='status':
        import json
        from .engine import completed
        result={}
        for role in ['primary','control']:
            stages=list((args.run_dir/'production'/role).glob('*/rep_*'))
            result[role]=sum(completed(p) is not None for p in stages)
        result['quarantined_preparations']=[str(p.relative_to(args.run_dir)) for p in (args.run_dir/'preparations').glob('*/hold_*/QUARANTINE.json')]
        result['failed_attempts']=len(list(args.run_dir.rglob('FAILED.json')))
        result['smoke_qualified']=(args.run_dir/'QUALIFIED.json').exists()
        print(json.dumps(result,indent=2))
    else:
        from .analysis import analyze
        analyze(args)

if __name__=='__main__': main()
