
import jax
from jax import grad, jit, vmap,lax
import jax.numpy as jnp
import numpy as np
from functools import partial

from timeit import default_timer as timer
from datetime import datetime
import re


class DNN:
    def __init__(self, unitName, N, M, p, Omega,Lant_dim = 0,delta =1.0):
        self.N = N
        self.M = M
        self.Lan = Lant_dim
        self.p  = 2*p + self.Lan
        p = 2*p + self.Lan
        self.unitfun = unittanh
        self.knots = lambda Z: []
        self.unitIsPeriodic = 0
        curUfunScalar = ufunScalarHelper
        print('p:',self.p)
        self.ufunScalar = lambda x, cHat: curUfunScalar(N, M, self.p, self.unitfun, x, cHat)
        self.ufun = lambda x, cHat: vmap(curUfunScalar, in_axes = (None, None, None, None, 0, None), out_axes = 0)(N, M, self.p,self.unitfun, jnp.atleast_1d(x), cHat)
        self.paramShape = (N, self.p+2+(M-1)*(N+1))
        self.embdfun = partial(PeriodEmbding, Lp = jnp.pi/delta)
        curfunLatent = ufunScalarLatent
        self.ufunLatentScalar = lambda x,latent,cHat:curfunLatent(self.N,self.M,self.p,self.embdfun,self.unitfun,x,latent,cHat)
        self.ufunLatent = lambda x,latent,cHat:vmap(curfunLatent,in_axes=(None,None,None,None,None,0,None,None),out_axes=0)(self.N,self.M,self.p,self.embdfun,self.unitfun,jnp.atleast_1d(x),latent,cHat)
        self.ufunLatentScalarxy = jit(lambda x,y,latent,cHat:curfunLatent(self.N,self.M,self.p,self.embdfun,self.unitfun,jnp.hstack([x,y]),latent,cHat))
        self.ufunWithLatentScalar = lambda x,Hat_Lantent:curfunLatent(self.N,self.M,self.p,self.embdfun,self.unitfun,x,Hat_Lantent[:self.Lan],Hat_Lantent[self.Lan:])
        self.ufunWithLatent = lambda x,Hat_Lantent:vmap(curfunLatent,in_axes=(None,None,None,None,None,0,None,None),out_axes=0)(self.N,self.M,self.p,self.embdfun,self.unitfun,jnp.atleast_1d(x),Hat_Lantent[:self.Lan],Hat_Lantent[self.Lan:])



    def getInitAZ(self, key, Omega):
        key, subkey = jax.random.split(key)
        azInit = jax.random.normal(subkey, shape=(self.paramShape[0]*self.paramShape[1]+1,))
        return azInit, key

def ufunScalarHelper(N, M, p, unitfun, x, cHat):
    hHat = cHat[-(N+1):].reshape((1,-1))
    cHat = cHat[:-(N+1)].reshape((N,-1))
    Lp = jnp.pi
    x = jnp.hstack((jnp.sin(Lp *x),jnp.cos(Lp * x)))
    y = unitfun(x, cHat[:, 0:p+1])
    y = unitfun(y, cHat[:, p+1:p+1+(N+1)])
    y = unitfun(y, cHat[:, p+1+(N+1):p+1+2*(N+1)])
    y = Linear(y, hHat).reshape(())
    return y


def ufunScalarLatent(N, M, p, embdfun,unitfun, x, latent, cHat):
    hHat = cHat[-(N+1):].reshape((1,-1))
    cHat = cHat[:-(N+1)].reshape((N,-1))
    required = (p + 1) + (M-1)*(N+1)
    x_embding = embdfun(x,latent)
    y = unitfun(x_embding, cHat[:, 0:p+1])
    y = Linear(y, hHat).reshape(())
    return y


@jit
def Linear(x,Z):
    print('compile unit LInar')
    return (jnp.dot(Z[:, :-1], x.reshape((-1,1))) + Z[:, -1].reshape((-1,1)))

@jit
def unittanh(x, Z):

    print('compile unit tanh')
    print("Z.shape:",Z.shape)
    print("x.shape:",x.shape)
    return jnp.tanh(jnp.dot(Z[:, :-1], x.reshape((-1,1))) + Z[:, -1].reshape((-1,1)))

@jit
def IdvEmbding(x,z_hid,Lp=1.0):
    print("Compile Idv Embding")
    return jnp.hstack((x, z_hid.reshape(-1)))


@jit
def PeriodEmbding(x,z_hid,Lp= 1.0):
    print("Compile Period Embding")
    print("x.shape:",x.shape,"z_hid.shape:",z_hid.shape)
    return jnp.hstack((jnp.sin(Lp *x),jnp.cos(Lp * x), z_hid.reshape(-1)))

@jit
def PeriodE(x,z_hid,Lp= 1.0):
    print("Compile Period Embding")
    return jnp.hstack((jnp.sin(Lp *x),jnp.cos(Lp * x)))
