"""Proper finite-support completion of Hummer's uniform-linear-rate posterior.

The declared measure is Lebesgue in relative free energies at a fixed anchor
and independent positive forward rates. No log-rate prior or smoothing is used.
"""
import numpy as np
import hummer_original_bayesian as H
import hummer_high_precision as P
from scipy.linalg import expm


def relative_to_last(g,anchor,n):
    full=np.zeros(n);full[np.arange(n)!=anchor]=g
    return full[:-1]-full[-1]


def log_target(g,k,counts,tau,G,K,anchor):
    if (np.any(np.abs(g)>G) or np.any(k<=0) or np.any(k>=K)
            or not np.isfinite(g).all() or not np.isfinite(k).all()):
        return -np.inf
    return P.evaluate(relative_to_last(g,anchor,len(counts)),k,counts,tau,precision='quad')


def observables(g,k,anchor,n,dx):
    transformed=relative_to_last(g,anchor,n)
    d,f=H.observables(transformed,k,dx)
    return d,f-f[anchor]


def surrogate(g,k,counts,tau,anchor):
    """Positive cheap screening density, never the posterior target.

    Clip only surrogate free energies to tame stiff float64 matrices. Exact
    second-stage acceptance restores the declared unclipped bounded posterior.
    Probability floors here ensure screening support everywhere the target is
    positive; they never enter the high-precision likelihood or stored samples.
    """
    p=H.populations(relative_to_last(np.clip(g,-10.,10.),anchor,len(counts)))
    approximate=H.log_target(relative_to_last(np.clip(g,-10.,10.),anchor,len(counts)),k,counts,tau)
    if np.isfinite(approximate):
        return approximate
    with np.errstate(over='ignore',invalid='ignore'):
        t=expm(tau*H.generator(p,k))
    if not np.isfinite(t).all():
        t=np.broadcast_to(p[:,None],counts.shape).copy()
    t=np.maximum(t,1e-300);t/=t.sum(axis=0)[None,:]
    use=counts>0
    return float(np.sum(counts[use]*np.log(t[use])))


def sample(counts,centers,tau,g0,k0,G,K,anchor,steps=180000,burn=60000,
           thin=20,seed=123,progress=None,global_probability=.25,proposal_scales=None):
    """Metropolis mixture, invariant for a *uniform-linear-rate* finite box.

    Both local additive uniform moves and full-prior coordinate draws are
    symmetric. Out-of-box local proposals are rejected without redraw. Scale
    adaptation is restricted to warm-up. Global moves improve exploration of
    broad flat-prior tails without changing the target measure.
    """
    counts=np.asarray(counts);centers=np.asarray(centers);n=len(centers);m=n-1
    if not 0<=burn<steps or G<=0 or K<=0 or thin<1 or not 0<=anchor<n:
        raise ValueError('Invalid finite support or chain schedule')
    dx=float(np.diff(centers)[0])
    if not np.allclose(np.diff(centers),dx):
        raise ValueError('Uniform bins required')
    g=np.asarray(g0,dtype=float).copy();k=np.asarray(k0,dtype=float).copy()
    if len(g)!=m or len(k)!=m:
        raise ValueError('Incorrect state dimensions')
    ll=log_target(g,k,counts,tau,G,K,anchor)
    screen=surrogate(g,k,counts,tau,anchor)
    if not np.isfinite(ll):
        raise ValueError('Invalid initial state')
    rng=np.random.default_rng(seed)
    P.reset_statistics()
    scale=(np.r_[np.full(m,.25),np.maximum(.3*k,.05)] if proposal_scales is None
           else np.asarray(proposal_scales,dtype=float).copy())
    if scale.shape!=(2*m,) or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Invalid proposal scales')
    accepted=np.zeros(2*m);tried=np.zeros(2*m)
    total_accepted=np.zeros((2,2*m));total_tried=np.zeros((2,2*m))
    numerical_rejections=np.zeros(2*m,dtype=int);boundary_rejections=np.zeros(2*m,dtype=int)
    ds=[];fs=[];gs=[];ks=[];ls=[]
    rejected_examples=[]
    stage1_rejections=0
    for step in range(steps):
        index=int(rng.integers(2*m));global_move=rng.random()<global_probability
        new_g=g.copy();new_k=k.copy()
        if index<m:
            new_g[index]=rng.uniform(-G,G) if global_move else new_g[index]+rng.uniform(-scale[index],scale[index])
        else:
            j=index-m
            new_k[j]=rng.uniform(0.,K) if global_move else new_k[j]+rng.uniform(-scale[index],scale[index])
        outside=np.any(np.abs(new_g)>G) or np.any(new_k<=0) or np.any(new_k>=K)
        if outside:
            proposal=-np.inf;new_screen=-np.inf;boundary_rejections[index]+=1
        else:
            new_screen=surrogate(new_g,new_k,counts,tau,anchor)
            if np.log(rng.random())<min(0.,new_screen-screen):
                # MPFR failures raise and abort; they are not silent rejections.
                proposal=log_target(new_g,new_k,counts,tau,G,K,anchor)
            else:
                proposal=-np.inf;stage1_rejections+=1
        mode=int(global_move);total_tried[mode,index]+=1
        if not global_move:
            tried[index]+=1
        if np.isfinite(proposal) and np.log(rng.random())<min(0.,proposal-ll-new_screen+screen):
            g,k,ll,screen=new_g,new_k,proposal,new_screen
            total_accepted[mode,index]+=1
            if not global_move:
                accepted[index]+=1
        if step<burn and (step+1)%1000==0:
            fraction=np.divide(accepted,tried,out=np.full(2*m,.35),where=tried>0)
            scale*=np.exp(np.clip(fraction-.35,-.25,.25))
            # Proposal widths are tuning, not parameter restrictions.
            scale[:m]=np.minimum(scale[:m],2*G);scale[m:]=np.minimum(scale[m:],K)
            tried[:]=0;accepted[:]=0
        if step>=burn and (step-burn)%thin==0:
            d,f=observables(g,k,anchor,n,dx)
            ds.append(d);fs.append(f);gs.append(g.copy());ks.append(k.copy());ls.append(ll)
        if progress and (step+1)%50000==0:
            progress(step+1,ll)
    values=dict(D=np.asarray(ds),F=np.asarray(fs),g=np.asarray(gs),k=np.asarray(ks),log_likelihood=np.asarray(ls),
                acceptance=np.divide(total_accepted,total_tried,out=np.zeros_like(total_accepted),where=total_tried>0),
                numerical_rejections=numerical_rejections,boundary_rejections=boundary_rejections,
                numerical_rejection_examples=np.asarray(rejected_examples),final_proposal_scales=scale)
    values.update({key:np.array(value) for key,value in P.statistics().items()})
    values['stage1_rejections']=np.array(stage1_rejections)
    if any(not np.isfinite(values[key]).all() for key in ['D','F','g','k','log_likelihood']):
        raise FloatingPointError('Nonfinite retained chain')
    if np.any(values['log_likelihood']>1e-8):
        raise FloatingPointError('Invalid positive log probability')
    return values
