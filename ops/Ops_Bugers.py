import jax
from jax import grad, jit, vmap, jvp, value_and_grad
import jax.numpy as jnp
from functools import partial
from jax import lax 

class OpsBurgers:
    def __init__(self, prob, dnn, scheme, modeName):
        self.prob = prob   
        self.dnn = dnn    
        self.modeName = modeName   
        if(self.prob.OmegaPeriodic == 1 and self.dnn.unitIsPeriodic == 1):
            print("No boundary penality because units satisfy boundary conditions")
            self.bc = lambda xdfun: 0.   
        else:
            print("Enforcing boundary conditions via penalty")
            self.bc = self.prob.bc       
        if(scheme == "RK23" or scheme == "RK45"):
            if(modeName == 'NGLM'):
                self.rhsJ = self.rhsJF2  
            elif(modeName == 'NGG'):
                self.rhsJ = self.rhsJF1
            elif(modeName == 'NGW'):
                self.rhsJ = self.rhsJF3
            else:
                raise Exception('Not implemented')
        else:
            raise Exception("not implemented")

    
    def rhsJF1(self, x, alphaZ, t):
        return BurgersRHS(self.dnn.ufunScalar, self.dnn.ufun, self.bc, self.prob.v, x.reshape((-1,)), alphaZ, t)
    
    def rhsJF2(self, x, alphaZ, Latent,delta, t,key):
        return BurgersRHS_Latent(self.dnn.ufunLatentScalar, self.dnn.ufunLatent, self.bc, self.prob.v, x.reshape((-1,)), alphaZ,Latent,delta, t,key)

    def rhsJF3(self, x, Latent_alphaZ, t):
        return BurgersRHS(self.dnn.ufunWithLatentScalar, self.dnn.ufunWithLatent, self.bc, self.prob.v, x.reshape((-1,)), Latent_alphaZ, t) 


@partial(jit, static_argnums=(0,1,2,))
def BurgersRHS_Latent(ufunLantentScalar,ufunLatent,bc,v,x,alphaZ,Latent,delta,t,key):
    print('compile rhs Latents') 
    dx = jax.vmap(grad(ufunLantentScalar,argnums=0),in_axes=(0,None,None,None),out_axes=0)(x,Latent,delta,alphaZ)
    dxx = jax.vmap(grad(grad(ufunLantentScalar,argnums=0),argnums=0),in_axes=(0,None,None,None),out_axes=0)(x,Latent,delta,alphaZ)
    Jac = jax.jacfwd(lambda az:ufunLatent(x,Latent,delta,az))(alphaZ)
    S_t = jax.random.choice(key, len(alphaZ), shape=(1500,), replace=False) # create random indices over the columns
    Jac_sub = jnp.take(Jac, S_t, axis=1)  # subsample columns
    u = ufunLatent(x,Latent,delta,alphaZ) 
    f = jnp.multiply(u, dx) - v*dxx
    J_sub = jnp.linalg.lstsq(Jac_sub,-f,rcond=-1)[0]  
    J = jnp.zeros(len(alphaZ)).at[S_t].set(J_sub)
    
    return J 




