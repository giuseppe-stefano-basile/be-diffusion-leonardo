"""Native Stochkin passage calculation and unchanged diffusion reconstruction."""
import csv
import inspect
from pathlib import Path
import sys
import numpy as np
from .common import REPO
sys.path.insert(0,str(REPO/'vendor'))
from native_fpe import mfpt_1d_smolu_integral as native_integral
from native_reconstruction import interface_to_centers,interpolate_D_to_grid
from diffusion_models import SplineLogDiffusion1D
from diagnostics import rank_diagnostics

# Concrete dtype specialization only; native algorithm/order retained, no fastmath.
_namespace={'np':np}
exec(compile(inspect.getsource(native_integral).replace('dtype=float','dtype=np.float64'),'<native-stochkin-float64>','exec'),_namespace)
try:
    import numba
    integral=numba.njit(cache=False,fastmath=False)(_namespace['mfpt_1d_smolu_integral'])
except ImportError:
    integral=native_integral
R=.008314462618


class Smoother:
    def __init__(self,x,h=.1):
        self.centers,_,_=interface_to_centers(x,np.ones(len(x)))
        self.model=SplineLogDiffusion1D(self.centers[0],self.centers[-1],n_basis=len(self.centers),D_min=1e-300)
        grid=np.linspace(*self.centers[[0,-1]],801)
        B=self.model.basis_matrix(self.centers); B2=self.model.basis_matrix(grid,derivative=2)
        integrate=getattr(np,'trapezoid',None) or np.trapz
        penalty=integrate(B2[:,:,None]*B2[:,None,:],grid,axis=0)/(self.centers[-1]-self.centers[0])
        self.operator=np.linalg.solve(B.T@B/len(self.centers)+h**4*penalty,B.T/len(self.centers))
    def fit(self,x,D):
        _,dc,_=interface_to_centers(x,D,method='harmonic')
        return self.operator@np.log(dc)


def domain(fes,event,mode,n=4001,factor=1.,boundary=None):
    if mode=='association': lo,hi=(event['left'] if boundary is None else boundary),event['onset']
    else: lo,hi=event['dissociation'],(event['right'] if boundary is None else boundary)
    if not lo<event['source']<hi: raise ValueError('Source outside passage domain')
    s=np.linspace(lo,hi,n); F=np.interp(s,fes[:,0],fes[:,1]); mid=int(np.argmin(abs(s-event['source'])))
    good=F<=F.min()+factor*R*298; left=right=mid
    while left>0 and good[left-1]: left-=1
    while right<n-1 and good[right+1]: right+=1
    ids=np.arange(left,right+1); target=n-1 if mode=='association' else 0
    if target in ids: raise ValueError('Absorber overlaps source core')
    return dict(s=s,F=F,ids=ids,mid=mid,target=target,weights=np.exp(-(F[ids]-F[ids].min())/(R*298)))


class Passage:
    def __init__(self,x,fes,event,profile='raw_median',n=4001,factor=1.,boundaries=None):
        self.x=np.asarray(x); self.smoother=Smoother(x) if profile.startswith('smooth') else None
        self.domains={m:domain(fes,event,m,n,factor,(boundaries or {}).get(m)) for m in ['association','dissociation']}
        if self.smoother:
            for d in self.domains.values(): d['basis']=self.smoother.model.basis_matrix(d['s'])
    def evaluate(self,D,validate_native=False,exact=False):
        if not np.isfinite(D).all() or np.any(D<=0): raise ValueError('Invalid diffusion profile')
        centers,dc,_=interface_to_centers(self.x,D,method='harmonic')
        coefficients=self.smoother.fit(self.x,D) if self.smoother else None
        result={}
        for mode,d in self.domains.items():
            s,F,ids=d['s'],d['F'],d['ids']
            if s[0]<centers[0] or s[-1]>centers[-1]: raise ValueError('Diffusion extrapolation refused')
            dg=(np.clip(np.exp(np.clip(d['basis']@coefficients,-700.,700.)),1e-300,np.inf) if self.smoother
                else interpolate_D_to_grid(s,centers,dc,method='linear'))
            points=[int(ids[0]),d['mid'],int(ids[-1])]
            times=np.array([integral(s,F,dg,1/(R*298),j,d['target']) for j in points])
            if not np.isfinite(times).all() or np.any(times<=0): raise ValueError('Invalid MFPT')
            bound=float(np.ptp(times)/times[1]); use_exact=exact or bound>1e-8
            value=float(np.average([integral(s,F,dg,1/(R*298),int(j),d['target']) for j in ids],weights=d['weights'])) if use_exact else float(times[1])
            err=0.
            if validate_native:
                err=max(abs(t/native_integral(s,F,dg,1/(R*298),j,d['target'])-1) for j,t in zip(points,times))
                if err>1e-12: raise ValueError('Native Stochkin specialization differs')
            if integral(s,F,dg,1/(R*298),d['target'],d['target'])!=0: raise ValueError('Absorbing MFPT is not zero')
            result[mode]=dict(MFPT_seconds=value*1e-12,rate_s_inverse=1e12/(4*value),source_core_relative_bound=bound,
                exact_core_average=use_exact,native_relative_error=err)
        return result


def diagnostics(values):
    values=np.asarray(values)
    if values.ndim==2: values=values[...,None]
    rh,ess=rank_diagnostics(values); tail=[]
    for q in [.05,.95]:
        _,e=rank_diagnostics((values<=np.quantile(values,q,axis=(0,1))).astype(float)); tail.append(e)
    te=np.minimum(*tail)
    # JSON uses null for undefined/nonfinite diagnostics, with pass=False.
    clean=lambda a:[float(x) if np.isfinite(x) else None for x in a]
    return dict(Rhat=clean(rh),bulk_ESS=clean(ess),tail_ESS=clean(te),passed=((rh<=1.01)&(ess>=400)&(te>=400)).tolist())


def baseline_regression(baseline):
    checks=[]
    for cv in ['R25','R27']:
        folder=Path(baseline)/cv
        with (folder/'D_all_bins.csv').open() as f: rows=list(csv.DictReader(f))
        x=np.array([float(r['s']) for r in rows]); D=np.array([float(r['median']) for r in rows])
        fes=np.loadtxt(folder/'fes.dat')
        from .common import read
        event=read(folder/'events.json')
        with (folder/'kinetics.csv').open() as f: reference=list(csv.DictReader(f))
        for profile in ['raw_median','smooth_median_h0p10']:
            result=Passage(x,fes,event,profile).evaluate(D,validate_native=True,exact=True)
            for mode,row in result.items():
                expected=next(float(r['per_water_rate_s_inverse']) for r in reference if r['profile']==profile and r['mode']==mode)
                error=abs(row['rate_s_inverse']/expected-1)
                if error>1e-10: raise ValueError(f'Baseline kinetics regression failed: {cv}/{profile}/{mode}: {error}')
                checks.append(dict(CV=cv,profile=profile,mode=mode,relative_error=error,rate_s_inverse=row['rate_s_inverse']))
    return checks
