"""Portable loader for the archived, unchanged quad/384-bit propagator."""
import ctypes
import ctypes.util
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import numpy as np
_libraries={}
_lock=threading.Lock()
_stats=threading.local()

def library(precision='mpfr'):
    if precision in _libraries: return _libraries[precision]
    with _lock:
        if precision in _libraries: return _libraries[precision]
        source=Path(__file__).with_suffix('.cpp')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        cache=Path(os.environ.get('HUMMER_CACHE_DIR',Path.home()/'.cache/be-diffusion'))
        cache.mkdir(parents=True,exist_ok=True)
        shared=cache/f'hummer_{precision}_{digest}.so'
        if not shared.exists():
            temporary=cache/f'.build_{precision}_{os.getpid()}.so'
            command=[os.environ.get('CXX','g++'),'-O3','-std=c++17','-shared','-fPIC']
            if precision=='quad': command+=['-DHUMMER_QUAD',str(source),'-lquadmath']
            else:
                candidates=[Path(os.environ.get('HUMMER_MPFR_INCLUDE','/usr/include')),Path(sys.prefix)/'include']
                if os.environ.get('CONDA_PREFIX'): candidates.append(Path(os.environ['CONDA_PREFIX'])/'include')
                include=next((p for p in candidates if (p/'mpfr.h').is_file()),None)
                if include is None: raise RuntimeError('MPFR headers missing: set HUMMER_MPFR_INCLUDE or load the MPFR development module')
                command+=['-I'+str(include),str(source)]
                if os.environ.get('HUMMER_MPFR_LIBDIR'): command+=['-L'+os.environ['HUMMER_MPFR_LIBDIR']]
                mpfr_library=ctypes.util.find_library('mpfr') or 'libmpfr.so.6'
                command+=[mpfr_library if Path(mpfr_library).is_absolute() else '-l:'+mpfr_library]
            command+=['-o',str(temporary)]
            r=subprocess.run(command,capture_output=True,text=True)
            if r.returncode: raise RuntimeError('High-precision build failed: '+r.stderr)
            os.replace(temporary,shared)
        loaded=ctypes.CDLL(str(shared)); pointer=ctypes.POINTER(ctypes.c_double)
        loaded.hummer_loglike_mp.argtypes=[ctypes.c_int,pointer,pointer,pointer,ctypes.c_double,pointer,pointer]
        loaded.hummer_loglike_mp.restype=ctypes.c_int
        _libraries[precision]=loaded
        return loaded

def evaluate(g,k,counts,tau,return_transition=False,precision='mpfr'):
    g=np.ascontiguousarray(g,dtype=np.float64); k=np.ascontiguousarray(k,dtype=np.float64)
    counts=np.ascontiguousarray(counts,dtype=np.float64); n=len(counts)
    if g.shape!=(n-1,) or k.shape!=(n-1,) or counts.shape!=(n,n): raise ValueError('Incorrect high precision shapes')
    ll=ctypes.c_double(); t=np.empty((n,n)) if return_transition else None
    def ptr(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    code=library(precision).hummer_loglike_mp(n,ptr(g),ptr(k),ptr(counts),float(tau),ctypes.byref(ll),ptr(t) if t is not None else None)
    if code:
        if precision=='quad':
            _stats.mpfr_fallbacks=getattr(_stats,'mpfr_fallbacks',0)+1
            return evaluate(g,k,counts,tau,return_transition,precision='mpfr')
        raise ArithmeticError(f'384-bit propagator failed, status {code}; chain aborted')
    if precision=='quad': _stats.quad_successes=getattr(_stats,'quad_successes',0)+1
    return (ll.value,t) if return_transition else ll.value

def reset_statistics(): _stats.quad_successes=0; _stats.mpfr_fallbacks=0

def statistics(): return dict(quad_successes=getattr(_stats,'quad_successes',0),mpfr_fallbacks=getattr(_stats,'mpfr_fallbacks',0))
