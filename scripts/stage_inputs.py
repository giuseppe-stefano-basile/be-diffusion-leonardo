#!/usr/bin/env python3
"""Validate and extract the external input archive without overwriting inputs."""
import argparse
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from be_pilot.common import REPO,read,sha,verify_manifest
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--destination',required=True,type=Path);a=p.parse_args()
expected=read(REPO/'config/input_bundle.json')['archive_sha256']
if sha(a.archive)!=expected: raise SystemExit('External archive checksum mismatch')
dest=a.destination.expanduser().resolve();dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists():
    verify_manifest(dest)
    if sha(dest/'SHA256SUMS.json')!=read(REPO/'config/input_bundle.json')['manifest_sha256']: raise SystemExit('Different inputs already staged')
    print('Verified existing inputs:',dest);raise SystemExit(0)
with tempfile.TemporaryDirectory(prefix='.be-inputs-',dir=dest.parent) as temp:
    root=Path(temp)
    with tarfile.open(a.archive,'r:gz') as archive:
        for m in archive.getmembers():
            relative=Path(m.name)
            if relative.is_absolute() or '..' in relative.parts or not (m.isfile() or m.isdir()): raise SystemExit('Unsafe archive member')
        archive.extractall(root)
    extracted=root/'inputs';verify_manifest(extracted)
    extracted.rename(dest)
print('Verified and staged:',dest)
