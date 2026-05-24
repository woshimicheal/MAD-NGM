import jax
from jax import grad, jit, vmap, value_and_grad, jvp
import jax.numpy as jnp
from functools import partial

class OpsKdV:
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
        return KDVRHS(self.dnn.ufunScalar, self.dnn.ufun, self.bc, self.prob.v, x.reshape((-1,)), alphaZ, t)
    
    def rhsJF2(self, x, alphaZ, Latent, t,key):
        return KDVRHS_Latent(self.dnn.ufunLatentScalar, self.dnn.ufunLatent, self.bc, self.prob.v, x.reshape((-1,)), alphaZ,Latent, t,key)

    def rhsJF3(self, x, Latent_alphaZ, t):
        return KDVRHS(self.dnn.ufunWithLatentScalar, self.dnn.ufunWithLatent, self.bc, self.prob.v, x.reshape((-1,)), Latent_alphaZ, t)


@partial(jit, static_argnums=(0,1,2,))
def KDVRHS_Latent(ufunLantentScalar,ufunLatent,bc,v,x,alphaZ,Latent,t,key):
    # print('compile rhs Latents')
    dx = jax.vmap(grad(ufunLantentScalar,argnums=0),in_axes=(0,None,None),out_axes=0)(x,Latent,alphaZ)
    dxxx = jax.vmap(grad(grad(grad(ufunLantentScalar, argnums = 0), argnums = 0), argnums = 0), in_axes = (0,None, None), out_axes = 0)(x,Latent, alphaZ) 
    Jac = jax.jacfwd(lambda az:ufunLatent(x,Latent,az))(alphaZ)
    u = ufunLatent(x,Latent,alphaZ) 
    residual = jnp.multiply(u, dx) + 0.0025*dxxx
    J = jnp.linalg.lstsq(Jac, -residual, rcond=-1)[0] 
    return J 

