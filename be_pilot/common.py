from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DT = .0005
SPECIES = ['Be'] + ['O', 'H', 'H'] * 125
TYPES = np.array([1] + [2, 3, 3] * 125)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, default=convert, allow_nan=False)
        f.write('\n')
        name = f.name
    os.replace(name, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed(*parts):
    return 1 + int(digest(parts)[:14], 16) % 899999999


def cn(xyz, box, r0):
    delta = xyz[1::3] - xyz[0]
    lengths = box[:, 1] - box[:, 0]
    delta -= lengths * np.rint(delta / lengths)
    return float(np.sum(1 / (1 + np.exp(4 * (np.linalg.norm(delta, axis=1) - r0)))))


def check_geometry(types, xyz, box, expected_box=None):
    if not np.array_equal(types, TYPES):
        raise ValueError('Expected ordered Be + 125 (O H H) atoms with types 1/2/3')
    if xyz.shape != (376, 3) or box.shape != (3, 2) or not np.isfinite(xyz).all() or not np.isfinite(box).all():
        raise ValueError('Malformed geometry')
    if np.any(box[:, 1] <= box[:, 0]): raise ValueError('Invalid periodic box')
    if expected_box is not None and not np.allclose(box, expected_box, rtol=0, atol=1e-9):
        raise ValueError('Fixed periodic box changed')


def read_data(path):
    lines = Path(path).read_text().splitlines()
    if any('xy xz yz' in x for x in lines): raise ValueError('Triclinic box not supported')
    box = np.array([list(map(float, next(x for x in lines if tag in x).split()[:2]))
                    for tag in ('xlo xhi', 'ylo yhi', 'zlo zhi')])
    start = next(i for i, x in enumerate(lines) if x.startswith('Atoms')) + 1
    rows = []
    for line in lines[start:]:
        p = line.split('#')[0].split()
        if not p: continue
        if not p[0].isdigit(): break
        rows.append(p)
    rows.sort(key=lambda p: int(p[0]))
    if [int(p[0]) for p in rows] != list(range(1, 377)): raise ValueError('Atom IDs changed')
    types = np.array([int(p[1]) for p in rows])
    xyz = np.array([list(map(float, p[2:5])) for p in rows])
    check_geometry(types, xyz, box)
    return xyz, box


def write_data(path, xyz, box):
    check_geometry(TYPES, xyz, box)
    xyz = box[:, 0] + (xyz - box[:, 0]) % (box[:, 1] - box[:, 0])
    lines = ['Be diffusion pilot: fixed original cell', '', '376 atoms', '3 atom types', '']
    lines += [f'{a:.17g} {b:.17g} {c}lo {c}hi' for (a, b), c in zip(box, 'xyz')]
    lines += ['', 'Masses', '', '1 9.0122 # Be', '2 15.999 # O', '3 1.008 # H', '', 'Atoms # atomic', '']
    lines += [f'{i} {t} {x:.17g} {y:.17g} {z:.17g}' for i, (t, (x, y, z)) in enumerate(zip(TYPES, xyz), 1)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def dump_frames(path):
    with Path(path).open() as f:
        while True:
            line = f.readline()
            if not line: return
            if line.strip() != 'ITEM: TIMESTEP': raise ValueError('Malformed dump timestep')
            step = int(f.readline())
            if f.readline().strip() != 'ITEM: NUMBER OF ATOMS': raise ValueError('Missing atom count')
            n = int(f.readline())
            if n != 376: raise ValueError('Wrong atom count')
            if f.readline().strip() != 'ITEM: BOX BOUNDS pp pp pp': raise ValueError('Expected orthorhombic periodic box')
            box = np.array([list(map(float, f.readline().split())) for _ in range(3)])
            fields = f.readline().split()[2:]
            if fields != ['id', 'type', 'x', 'y', 'z']: raise ValueError('Unexpected dump columns')
            a = np.array([list(map(float, f.readline().split())) for _ in range(n)])
            if a.shape != (376, 5) or not np.array_equal(a[:, 0], np.arange(1, 377)):
                raise ValueError('Incomplete or unordered dump frame')
            check_geometry(a[:, 1], a[:, 2:], box)
            yield step, a[:, 2:], box


def colvar(path):
    fields = None
    with Path(path).open() as f:
        for line in f:
            if line.startswith('#! FIELDS'): fields = line.split()[2:]; break
    if fields is None or not {'time', 'cn25', 'cn27'} <= set(fields):
        raise ValueError('Expected time, cn25 and cn27 fields')
    a = np.atleast_2d(np.loadtxt(path))
    if a.shape[1] != len(fields) or not np.isfinite(a).all(): raise ValueError('Malformed COLVAR')
    return a[:, [fields.index(x) for x in ('time', 'cn25', 'cn27')]]


def verify_manifest(root, filename='SHA256SUMS.json'):
    root = Path(root)
    manifest = read(root / filename)
    for relative, expected in manifest.items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root.resolve()): raise ValueError('Unsafe manifest path')
        if sha(p) != expected: raise ValueError('Checksum mismatch: ' + str(p))
    return digest(manifest)
