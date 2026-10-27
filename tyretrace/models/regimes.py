"""Constrained left-to-right Gaussian pace-regime HMM.

States are descriptive pace regimes, not independently verified tyre physics.
Each stint resets the chain. Forward predictions never consume later outcomes.
Two versus three states is selected by held-out sequence likelihood.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


def _sequence(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError('a nonempty finite one-dimensional sequence is required')
    return a


@dataclass
class PaceHMM:
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    iterations: int=0
    converged: bool=False

    def __post_init__(self):
        self.means=np.asarray(self.means,dtype=float)
        self.variances=np.asarray(self.variances,dtype=float)
        self.transition=np.asarray(self.transition,dtype=float)
        self.initial=np.asarray(self.initial,dtype=float)
        k=len(self.means)
        if (k not in (2,3) or self.variances.shape!=(k,) or self.transition.shape!=(k,k)
            or self.initial.shape!=(k,) or np.any(self.variances<=0)
            or not all(np.all(np.isfinite(a)) for a in (self.means,self.variances,self.transition,self.initial))
            or np.any(self.transition<0) or np.any(self.initial<0)
            or not np.allclose(self.transition.sum(axis=1),1) or not np.isclose(self.initial.sum(),1)):
            raise ValueError('invalid HMM parameters')
        allowed=np.eye(k,dtype=bool)|np.eye(k,k,1,dtype=bool)
        if np.any(self.transition[~allowed]!=0):
            raise ValueError('regime transitions must stay or move one state forward')

    def _messages(self,values):
        y=_sequence(values);k=len(self.means)
        emission=-.5*(np.log(2*np.pi*self.variances)[None,:]+(y[:,None]-self.means)**2/self.variances)
        with np.errstate(divide='ignore'):
            trans=np.log(self.transition);initial=np.log(self.initial)
        alpha=np.empty((len(y),k));alpha[0]=initial+emission[0]
        for i in range(1,len(y)):
            alpha[i]=emission[i]+logsumexp(alpha[i-1,:,None]+trans,axis=0)
        return emission,trans,alpha,float(logsumexp(alpha[-1]))

    def log_likelihood(self,values):return self._messages(values)[-1]

    def posterior(self,values,*,smooth=False):
        emission,trans,alpha,ll=self._messages(values)
        if not smooth:
            return np.exp(alpha-logsumexp(alpha,axis=1)[:,None])
        beta=np.zeros_like(alpha)
        for i in range(len(alpha)-2,-1,-1):
            beta[i]=logsumexp(trans+emission[i+1,None,:]+beta[i+1,None,:],axis=1)
        return np.exp(alpha+beta-ll)

    def future(self,filtered,steps):
        p=np.asarray(filtered,dtype=float)
        if p.shape!=self.initial.shape or np.any(p<0) or not np.isclose(p.sum(),1) or steps<1:
            raise ValueError('valid filtered state probabilities and positive steps required')
        out=[]
        for _ in range(steps):
            p=p@self.transition;out.append(p.copy())
        probs=np.asarray(out);mean=probs@self.means
        variance=probs@(self.variances+self.means**2)-mean**2
        return probs,mean,np.maximum(variance,0)

    @classmethod
    def fit(cls,sequences,*,states=3,max_iter=50,tol=1e-5):
        ys=[_sequence(s) for s in sequences]
        if not ys or states not in (2,3):raise ValueError('sequences and two or three states required')
        joined=np.concatenate(ys)
        means=np.quantile(joined,[.65,.25,.90] if states==3 else [.35,.85])
        variance=np.full(states,max(float(np.var(joined)),.01))
        transition=np.eye(states)*.90
        for j in range(states-1):transition[j,j+1]=.10
        transition[-1,-1]=1.
        initial=np.zeros(states);initial[0]=1.
        model=cls(means,variance,transition,initial)
        previous=-np.inf
        for iteration in range(max_iter):
            occup=np.zeros(states);sum_y=np.zeros(states);sum_y2=np.zeros(states)
            counts=np.zeros((states,states));ll_total=0.
            for y in ys:
                emission,trans,alpha,ll=model._messages(y);ll_total+=ll
                beta=np.zeros_like(alpha)
                for t in range(len(y)-2,-1,-1):
                    beta[t]=logsumexp(trans+emission[t+1,None,:]+beta[t+1,None,:],axis=1)
                gamma=np.exp(alpha+beta-ll)
                occup+=gamma.sum(axis=0);sum_y+=gamma.T@y;sum_y2+=gamma.T@(y*y)
                for t in range(len(y)-1):
                    counts+=np.exp(alpha[t,:,None]+trans+emission[t+1,None,:]+beta[t+1,None,:]-ll)
            if iteration and abs(ll_total-previous)<tol*(1+abs(previous)):
                model.converged=True;break
            previous=ll_total
            model.means=np.where(occup>1e-6,sum_y/np.maximum(occup,1e-6),model.means)
            model.variances=np.maximum(sum_y2/np.maximum(occup,1e-6)-model.means**2,.0025)
            for j in range(states-1):
                row=counts[j,j:j+2]+np.array([1.,.1])
                model.transition[j,j:j+2]=row/row.sum()
            model.iterations=iteration+1
        return model


def select_regimes(sequences,*,max_iter=35):
    ys=[_sequence(s) for s in sequences if len(s)>=8]
    if len(ys)<3:raise ValueError('at least three practice runs of eight laps are required')
    train=[s[:max(4,int(.7*len(s)))] for s in ys]
    scores={}
    for k in (2,3):
        m=PaceHMM.fit(train,states=k,max_iter=max_iter)
        scores[k]=float(sum(m.log_likelihood(s)-m.log_likelihood(t) for s,t in zip(ys,train))
                        /sum(len(s)-len(t) for s,t in zip(ys,train)))
    chosen=max(scores,key=scores.get)
    return PaceHMM.fit(ys,states=chosen,max_iter=max_iter),scores
