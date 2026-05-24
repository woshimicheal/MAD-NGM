import jax
from jax import grad, jit, vmap, jvp, value_and_grad
import jax.numpy as jnp
from functools import partial

class OpsHeat:
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

    def rhsJF1(self, x, alphaZ, t,key):
        return HeatRHS(self.dnn.ufunScalar, self.dnn.ufun, self.bc, self.prob.v, x, alphaZ, t,key)
    
    def rhsJF2(self, x, alphaZ, Latent, t,key):
        return HeatRHS_Latent(self.dnn.ufunLatentScalar, self.dnn.ufunLatent, self.bc, self.prob.v, x, alphaZ,Latent, t,key)

    def rhsJF3(self, x, Latent_alphaZ, t,key):
        return HeatRHS(self.dnn.ufunWithLatentScalar, self.dnn.ufunWithLatent, self.bc, self.prob.v, x, Latent_alphaZ, t,key)


@partial(jit, static_argnums=(0,1,2,))
def HeatRHS_Latent(ufunLantentScalar,ufunLatent,bc,v,x,alphaZ,Latent,t,key):
    print('compile rhs Latents')
    ufunnn = jit(lambda x:ufunLantentScalar(x,Latent,0,alphaZ))
    hessian = jax.vmap(jax.jacfwd(grad(ufunnn)),in_axes=0,out_axes=0)(x)
    u_xx = hessian[:,0,0]
    u_yy = hessian[:,1,1]

    Jac = jax.jacfwd(lambda az:ufunLatent(x,Latent,0,az))(alphaZ)
    S_t = jax.random.choice(key, len(alphaZ), shape=(1500,), replace=False) 
    Jac = jnp.take(Jac, S_t, axis=1)  
    u = ufunLatent(x,Latent,0,alphaZ)
    f = 2*(u**3-u)- 0.001*(u_xx+u_yy) 
    J = jnp.linalg.lstsq(Jac,-f,rcond=-1)[0]
    J = jnp.zeros(len(alphaZ)).at[S_t].set(J)
    return J 



