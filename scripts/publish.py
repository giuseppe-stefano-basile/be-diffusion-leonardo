#!/usr/bin/env python3
"""Create/check the requested private repo, then push the local commit and tag.

Authentication stays in the process environment or GitHub CLI credential store.
No token is embedded in git URLs, files, or command-line arguments.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
OWNER='giuseppe-stefano-basile'; NAME='be-diffusion-leonardo'
ROOT=Path(__file__).resolve().parents[1]
token=os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
if not token and shutil.which('gh'):
    r=subprocess.run(['gh','auth','token'],capture_output=True,text=True)
    if r.returncode==0:token=r.stdout.strip()
if not token: raise SystemExit('No GitHub authentication available. Run gh auth login, or provide GH_TOKEN in your environment, then retry. Do not paste tokens into chat.')
def api(path,method='GET',data=None):
    request=urllib.request.Request('https://api.github.com'+path,method=method,
        headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'},
        data=json.dumps(data).encode() if data is not None else None)
    with urllib.request.urlopen(request) as r:return json.load(r)
# A source archive may not yet have Git metadata; initialize only this directory.
r=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,capture_output=True,text=True)
if r.returncode or Path(r.stdout.strip()).resolve()!=ROOT:
    subprocess.run(['git','init','-b','main',str(ROOT)],check=True)
for key,value in [('user.name',OWNER),('user.email',OWNER+'@users.noreply.github.com')]:
    r=subprocess.run(['git','config','--get',key],cwd=ROOT,capture_output=True,text=True)
    if r.returncode: subprocess.run(['git','config',key,value],cwd=ROOT,check=True)
sys_path=str(ROOT)
import sys
sys.path.insert(0,sys_path)
from be_pilot.common import read,verify_manifest
verify_manifest(ROOT)
allowed=set(read(ROOT/'SHA256SUMS.json'))|{'SHA256SUMS.json'}
tracked=subprocess.run(['git','ls-files'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines()
if set(tracked)-allowed: raise SystemExit('Unexpected tracked files: preserve these separately before publishing this handoff')
subprocess.run(['git','add','--',*sorted(allowed)],cwd=ROOT,check=True)
r=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
if r.returncode: subprocess.run(['git','commit','-m','Prepare verified R25/R27 diffusion pilot for Leonardo Booster'],cwd=ROOT,check=True)
r=subprocess.run(['git','rev-parse','--verify','refs/tags/v1.0.0'],cwd=ROOT,capture_output=True,text=True)
head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
if r.returncode: subprocess.run(['git','tag','v1.0.0'],cwd=ROOT,check=True)
elif r.stdout.strip()!=head: raise SystemExit('v1.0.0 already identifies another commit; preserving it')
if api('/user')['login'].lower()!=OWNER: raise SystemExit('Authenticate as '+OWNER+' to publish this personal repository')
try: repo=api('/repos/'+OWNER+'/'+NAME)
except urllib.error.HTTPError as e:
    if e.code!=404: raise
    repo=api('/user/repos','POST',dict(name=NAME,private=True,description='Reproducible Be R25/R27 diffusion pilot for Leonardo Booster',auto_init=False))
if not repo['private']:raise SystemExit('Refusing to push scientific inputs to a public repository')
url='https://github.com/'+OWNER+'/'+NAME+'.git'
r=subprocess.run(['git','remote','get-url','origin'],cwd=ROOT,capture_output=True,text=True)
if r.returncode:
    subprocess.run(['git','remote','add','origin',url],cwd=ROOT,check=True)
elif r.stdout.strip()!=url:raise SystemExit('Existing origin differs from requested repository; preserving it')
with tempfile.TemporaryDirectory(prefix='be-git-auth-') as temp:
    helper=Path(temp)/'askpass'
    helper.write_text('#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "x-access-token";; *) printf "%s\\n" "$BE_GIT_TOKEN";; esac\n')
    helper.chmod(0o700)
    env=os.environ.copy();env.update(BE_GIT_TOKEN=token,GIT_ASKPASS=str(helper),GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git','-c','credential.helper=','push','--atomic','-u','origin','main','refs/tags/v1.0.0'],cwd=ROOT,env=env,check=True)
print('Published private repository: https://github.com/'+OWNER+'/'+NAME)
