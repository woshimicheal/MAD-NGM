import sys
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import torch
pi = torch.tensor(np.pi)
from collections import OrderedDict
from scipy.io import loadmat
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
np.random.seed(2345)

class Net(torch.nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x ,selected_latent=None):
        if selected_latent!=None:
            concatenated_inputs = torch.cat((torch.sin(pi * x), torch.cos(pi * x), selected_latent.expand(x.size(0), -1)), dim=1).float().to(device)
        else:
            concatenated_inputs = torch.cat((torch.sin(pi * x),torch.cos(pi * x)), dim=1).float().to(device)
        out = self.layers(concatenated_inputs)
        return out




class MadNN():
    def __init__(self, u0Mat,xx0,layers,latent_size,num_latents,maxiter,model,para_dict=None,Latent =None):
        self.xx0 = torch.tensor(xx0,requires_grad=True).float().to(device)
        self.u0Mat = u0Mat.float().to(device)
        self.num_latents = num_latents
        self.latent_size = latent_size
        self.layers = layers

        if model == 'pretrain':
            self.Latents_pretrain()
        else:
            self.Latents_Modefine(para_dict,Latent)

        self.optimizer = torch.optim.LBFGS(self.net.parameters(),lr=0.01, max_iter=maxiter, max_eval=None, tolerance_grad=1e-30, tolerance_change=1e-30, history_size=1000, line_search_fn=None)
        self.mse_loss = torch.nn.MSELoss()
        self.losslist=[]
        self.optimizer_Adam = torch.optim.Adam(self.net.parameters())
        self.iter = 0

    def Latents_Modefine(self,para_dict,latent):

        net = Net(self.layers)
        net.load_state_dict(para_dict)
        for param in net.parameters():
            param.requires_grad = False
        latent = torch.nn.Parameter(latent)
        net.register_parameter('Latents', latent)
        self.net = net.to(device)

    def Latents_pretrain(self):
        net = Net(self.layers)
        latents = torch.nn.Parameter(torch.randn(self.num_latents, self.latent_size).float())
        net.register_parameter('Latents', latents)
        self.net = net.to(device)

    def net_u(self,X,index=None):
        selected_latent = self.net.Latents[index]
        u = self.net.forward(X,selected_latent).to(device)
        return u


    def loss_func_Helper(self,index=None):
        u_pred = self.net_u(self.xx0,index)
        uo_ii  = self.u0Mat[index].unsqueeze(-1).float().to(device)
        loss1 = torch.mean((uo_ii- u_pred) ** 2)
        return loss1

    def loss_func(self):

        self.optimizer.zero_grad()
        loss = 0
        Rular = 0.05
        for i in range(self.num_latents):
            loss += self.loss_func_Helper(i)+ Rular*torch.mean(self.net.Latents[i]**2)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0 or self.iter==0:
            print("it:%d,loss %.3e"%(self.iter,loss.item()))
            self.losslist.append((self.iter,loss.item()))
        return loss

    def train(self):
        self.net.train()
        loss = self.optimizer.step(self.loss_func)
        return loss

    def save_loss_to_file(self, file_path):
        with open(file_path, 'w') as file:
            for iter_num, loss_value in self.losslist:
                file.write(f"Iteration {iter_num}: {loss_value:.3e}\n")


def heat_loadData(file_load):
    data = loadmat(file_load)
    xx = data['points']
    tt = data['tspan']
    Input = data['input']
    xx_test = xx
    tt_test = tt
    Input_test = Input
    xx1  =  xx_test[0]
    xx2  =  xx_test[-1]
    xx0  =  xx_test
    u0Mat = Input_test
    return u0Mat,xx



def Mad_LoadData(file_load,s =200 ):

    data = loadmat(file_load)
    xx = data['xx'][:,:-1]
    tt = data['tspan']
    Input = data['input']
    Outpu = data['output']
    sub = 2
    xx_test = xx[:,::sub]
    tt_test = tt[:,::2]
    Input_test = Input[:-5,::sub]
    Outpu_test = Outpu[:-5,::2,::sub]
    xx_test = xx_test.flatten()[:,None]
    xx1  =  xx_test[0]
    xx2  =  xx_test[-1]
    xx0  =  xx_test
    u0Mat = Input_test
    return u0Mat,xx0,xx1,xx2

def Torch_jax_gai(path_torch,n=1):

    para_re = torch.load(path_torch)
    concatenated_params_list = []
    for i in range(n):
        w_i = 'layers.layer_'+str(i)+'.weight'
        b_i = 'layers.layer_'+str(i)+'.bias'
        wi = para_re[w_i]
        bi = para_re[b_i].reshape(-1,1)
        wb_i = torch.cat((wi,bi),dim = 1)
        concatenated_params_list.append(wb_i)
    wb_concatenated = torch.cat(concatenated_params_list, dim=1)
    print(wb_concatenated.shape)
    wb_concatenated = wb_concatenated.reshape(-1)
    print(wb_concatenated.shape)
    w_i = 'layers.layer_'+str(n)+'.weight'
    b_i = 'layers.layer_'+str(n)+'.bias'
    wi = para_re[w_i]
    bi = para_re[b_i].reshape(-1,1)
    wb_i = torch.cat((wi,bi),dim = 1).reshape(-1)
    para_test = torch.cat((wb_concatenated,wb_i)).cpu().detach().numpy()
    initAZ_test = jnp.array(para_test)
    Latent_jax = jnp.array(para_re['Latents'].cpu().detach().numpy())
    return Latent_jax,initAZ_test

def Torch_Jax(path_torch,model = None):

    para_test = torch.load(path_torch)
    concatenated_params_list = []
    for i in range(2):
        w_i = 'layers.layer_'+str(i)+'.weight'
        b_i = 'layers.layer_'+str(i)+'.bias'
        wi = para_test[w_i]
        bi = para_test[b_i].reshape(-1,1)
        concatenated_params_list.extend(torch.cat((wi,bi),dim=1).reshape(-1,1))
    all_params = torch.cat(concatenated_params_list, dim=0).cpu().detach().numpy()
    Latent_jax = jnp.array(para_test['Latents'].cpu().detach().numpy())
    initAZ_test = jnp.array(all_params)

    return Latent_jax,initAZ_test

def Torch_Jaxww(path_torch):
    para_test = torch.load(path_torch)
    concatenated_params_list = []
    for i in range(3):
        w_i = 'layers.layer_'+str(i)+'.weight'
        b_i = 'layers.layer_'+str(i)+'.bias'
        wi = para_test[w_i]
        bi = para_test[b_i].reshape(-1,1)
        concatenated_params_list.extend(torch.cat((wi,bi),dim=1).reshape(-1,1))
    all_params = torch.cat(concatenated_params_list, dim=0).cpu().detach().numpy()
    initAZ_test = jnp.array(all_params)

    return initAZ_test
