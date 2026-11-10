"""Hummer NJP 7 (2005) 34, Eqs. 8--15, unsmoothed uniform-rate variant.

The measure is d(g_1-g_N)...d(g_{N-1}-g_N) dk_1...dk_{N-1}, k_i>0.
There are no finite parameter caps and no log-uniform rate prior. Additive,
symmetric rate proposals that cross zero are rejected, never redrawn.

IMPORTANT: a literal unbounded uniform-rate prior has an improper posterior
for finite transition data: the fast-mixing likelihood approaches a strictly
positive constant. Finite-chain output is consequently diagnostic and cannot
define posterior expectations or credible intervals. See propriety_audit().
The sampler is provided to reproduce the paper's stated Metropolis target,
not to conceal this mathematical obstruction with implicit numerical caps.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm
from scipy.special import logsumexp


def populations(g):
    values = np.r_[np.asarray(g,dtype=float),0.]
    return np.exp(-values-logsumexp(-values))


def generator(p,k):
    p=np.asarray(p,dtype=float);k=np.asarray(k,dtype=float)
    n=len(p)
    if k.shape!=(n-1,) or not np.isfinite(p).all() or not np.isfinite(k).all() or np.any(p<=0) or np.any(k<=0):
        raise ValueError('Need positive finite populations and rates')
    back=k*p[:-1]/p[1:]
    r=np.zeros((n,n));i=np.arange(n-1)
    r[i+1,i]=k;r[i,i+1]=back
    np.fill_diagonal(r,-r.sum(axis=0))
    if not np.isfinite(r).all():
        raise FloatingPointError('Nonfinite generator')
    return r


def propagator(p,k,tau,counts=None):
    """Paper's detailed-balance symmetric eigenvalue evaluation, Eq. 13ff.

    A mathematically equivalent scaling/squaring fallback handles cancellation
    of very small observed probabilities in the eigenvector reconstruction.
    No missing probability columns are replaced by identity columns.
    """
    if tau<=0:
        raise ValueError('Positive lag required')
    r=generator(p,k)
    diag=np.diag(r)
    off=np.asarray(k)*np.sqrt(p[:-1]/p[1:])
    eig,u=eigh_tridiagonal(diag,off,check_finite=False,lapack_driver='stev')
    if eig[-1]>1e-9*max(1.,float(abs(diag).max())):
        raise FloatingPointError('Invalid positive generator eigenvalue')
    eig=np.minimum(eig,0.)
    t=((u*np.exp(tau*eig))@u.T)*np.sqrt(p[:,None]/p[None,:])
    observed=(counts>0) if counts is not None else np.ones_like(t,dtype=bool)
    if (not np.isfinite(t).all() or np.any(t[observed]<=0) or t.min()<-1e-12
            or not np.allclose(t.sum(axis=0),1.,rtol=1e-8,atol=1e-10)):
        t=expm(tau*r)
    if (not np.isfinite(t).all() or np.any(t[observed]<=0) or t.min()<-1e-10
            or t.max()>1.+1e-10 or not np.allclose(t.sum(axis=0),1.,rtol=1e-8,atol=1e-10)):
        raise FloatingPointError('Unresolved propagator arithmetic')
    return t


def log_target(g,k,counts,tau):
    """Log density in *linear* positive rates; no Jacobian or rate penalty."""
    if not np.isfinite(g).all() or not np.isfinite(k).all() or np.any(k<=0):
        return -np.inf
    p=populations(g)
    try:
        t=propagator(p,k,tau,counts)
    except (ValueError,FloatingPointError,np.linalg.LinAlgError):
        return -np.inf
    use=counts>0
    value=float(np.sum(counts[use]*np.log(t[use])))
    return value if value<=1e-8 else -np.inf


def observables(g,k,dx):
    p=populations(g)
    f=-np.log(p/dx)
    d=dx**2*np.asarray(k)*np.sqrt(p[:-1]/p[1:])
    return d,f


def propriety_audit(counts):
    """Analytic necessary check; it is independent of finite-chain mixing.

    For any fixed positive P and a connected R0, exp(tau*a*R0) -> P 1^T.
    L -> product(P_i ** sum_j N_ij)>0. This holds on an open cone of rate
    ratios, so integrating L against unbounded Lebesgue rate measure diverges.
    Restricting relative g to any compact region does not remove that cone.
    """
    counts=np.asarray(counts,dtype=float)
    if counts.ndim!=2 or counts.shape[0]!=counts.shape[1] or len(counts)<2 or not np.isfinite(counts).all() or np.any(counts<0):
        raise ValueError('Need a finite nonnegative square count matrix')
    p=(counts.sum(axis=1)+1.)/(counts.sum()+len(counts))
    return dict(normalizable=False,prior='Unbounded uniform relative g and positive linear nearest-neighbor rates',
        limiting_populations=p.tolist(),stationary_limit_log_likelihood=float(np.dot(counts.sum(axis=1),np.log(p))),
        reason='Full-dimensional fast-mixing rate cones retain strictly positive likelihood; the integral over unbounded positive linear rates diverges.',
        consequence='Posterior means and credible intervals are undefined under this literal prior. Finite chains cannot establish a normalized posterior, regardless of R-hat/ESS.',
        remedy_requires_method_choice='Specify a proper finite-support or decaying rate prior, or explicitly choose an optional paper regularization and prove propriety; no such replacement is applied here.')


def sample(counts,centers,tau,g0,k0,steps=40000,burn=10000,thin=20,seed=123,progress=None):
    """Finite diagnostic of the literal Metropolis target, Section 2.4.

    Symmetric additive proposals in free g and linear k. Proposal scales adapt
    only during burn-in; all production proposals then use fixed scales.
    Fixing g_N=0 removes only the arbitrary common free-energy constant.
    """
    counts=np.asarray(counts,dtype=float);centers=np.asarray(centers,dtype=float)
    if not 0<=burn<steps or thin<1 or len(centers)!=len(counts):
        raise ValueError('Invalid chain dimensions or schedule')
    delta=np.diff(centers)
    if not np.allclose(delta,delta[0]) or delta[0]<=0:
        raise ValueError('Paper Eq. 9 implementation requires a uniform spatial grid')
    rng=np.random.default_rng(seed);m=len(centers)-1
    g=np.asarray(g0,dtype=float).copy();k=np.asarray(k0,dtype=float).copy()
    ll=log_target(g,k,counts,tau)
    if not np.isfinite(ll):
        raise ValueError('Invalid initial state')
    scale=np.r_[np.full(m,.2),np.maximum(.2*k,.01)]
    tried=np.zeros(2*m,dtype=int);accepted=np.zeros(2*m,dtype=int)
    total_tried=np.zeros(2*m,dtype=int);total_accepted=np.zeros(2*m,dtype=int)
    ds=[];fs=[];gs=[];ks=[];ls=[]
    for step in range(steps):
        index=int(rng.integers(2*m));new_g=g.copy();new_k=k.copy()
        shift=rng.uniform(-scale[index],scale[index])
        if index<m:
            new_g[index]+=shift
        else:
            new_k[index-m]+=shift
        proposal=log_target(new_g,new_k,counts,tau)
        tried[index]+=1;total_tried[index]+=1
        if np.isfinite(proposal) and np.log(rng.random())<min(0.,proposal-ll):
            g,k,ll=new_g,new_k,proposal
            accepted[index]+=1;total_accepted[index]+=1
        if step<burn and (step+1)%1000==0:
            fraction=np.divide(accepted,tried,out=np.full(2*m,.35),where=tried>0)
            scale*=np.exp(np.clip(fraction-.35,-.25,.25))
            tried[:]=0;accepted[:]=0
        if step>=burn and (step-burn)%thin==0:
            d,f=observables(g,k,float(delta[0]))
            ds.append(d);fs.append(f);gs.append(g.copy());ks.append(k.copy());ls.append(ll)
        if progress and (step+1)%10000==0:
            progress(step+1,ll,float(k.max()))
    if not np.isfinite(ls).all() or np.any(np.asarray(ls)>1e-8):
        raise FloatingPointError('Invalid finite-chain likelihoods')
    return dict(D=np.asarray(ds),F=np.asarray(fs),g=np.asarray(gs),k=np.asarray(ks),log_likelihood=np.asarray(ls),
                acceptance=np.divide(total_accepted,total_tried,out=np.zeros(2*m),where=total_tried>0),
                final_proposal_scales=scale,
                diagnostic_only=True,prior='Uniform relative free energies and positive LINEAR rates; no bounds, no smoothness')
