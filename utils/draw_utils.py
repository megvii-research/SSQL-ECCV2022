import torch
import torch.nn.functional as F
from torch import nn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

__all__ = ['range_draw', 'hist_draw', 'sub_hist_draw']

#pre_path = '../save_fig/'
#model_name = 'pact_myGelu/'

def range_draw(x:torch.Tensor, dim:int, module_name:str):
    title = model_name+module_name+'_dim:'+str(dim)
    mpl.use('AGG')
    sns.set_style('darkgrid')
    with torch.no_grad():
        assert dim<=len(x.shape)
        tmp = x.clone().detach().cpu()
        max_x = tmp.transpose(0,dim).reshape(x.shape[dim],-1).max(dim=-1).values
        min_x = tmp.transpose(0,dim).reshape(x.shape[dim],-1).min(dim=-1).values
        r = max_x - min_x
        plt.plot(range(0, x.shape[dim]), max_x)
        plt.plot(range(0, x.shape[dim]), min_x)
        plt.legend(['max_val','min_val'])
        plt.title(title+ '_Min Max')
        plt.savefig(pre_path+title+ '_MinMax'+'.png')
        plt.close()

        plt.plot(range(0, x.shape[dim]), r)
        plt.legend(['range'])
        plt.title(title+ '_Range')
        plt.savefig(pre_path+title+ '_Range'+'.png')
        plt.close()


def sub_hist_draw(x:torch.Tensor, dim:int, sub_dim:int, module_name:str):
    title = model_name+module_name+'_dim:'+str(dim)+'_sub-dim: '+str(sub_dim)
    mpl.use('AGG')
    sns.set_style('darkgrid')
    with torch.no_grad():
        assert dim<=len(x.shape)
        assert sub_dim<=x.shape[dim]
        tmp = x.clone().detach().cpu()
        sub_dim_x = tmp.transpose(0,dim)[sub_dim].reshape(-1)
        fig = sns.displot(sub_dim_x)
        plt.title(title+ '_Hist')    
        fig.savefig(pre_path+title+ '_Hist'+'.png')
        plt.close()


def hist_draw(x:torch.Tensor, pre_path:str, method_name:str, module_name:str, xscale:str='linear', yscale:str='log' ,bins:int = 10000, binwidth:float = 0.01):
    title = module_name
    mpl.use('AGG')
    sns.set_style('darkgrid')
    with torch.no_grad():
        tmp = x.clone().detach().cpu().reshape(-1)
        #fig = sns.displot(tmp, bins = bins, binwidth = binwidth)
        fig = sns.displot(tmp)
        fig.ax.set_yscale(yscale)
        fig.ax.set_xscale(xscale)
        plt.title(title+ '_Hist')    
        fig.savefig(os.path.join(pre_path, method_name+'_'+title+ '_Hist'+'.png'))
        plt.close()


def hist_draw_new(x:torch.Tensor, pre_path:str, method_name:str, module_name:str, xscale:str='linear', yscale:str='log' ,bins:int = 10000, binwidth:float = 0.01):
    title = module_name
    mpl.use('AGG')
    sns.set_style('darkgrid')
    with torch.no_grad():
        tmp = x.clone().detach().cpu().reshape(-1)
        #fig = sns.displot(tmp, bins = bins, binwidth = binwidth)
        fig = sns.displot(tmp)
        fig.ax.set_yscale(yscale)
        fig.ax.set_xscale(xscale)
        #fig.ax.x()
        plt.tick_params(labelsize=8)
        #plt.title(title+ '_Hist')    
        fig.savefig(os.path.join(pre_path, method_name+'_'+title+ '_Hist'+'.png'), dpi=120)
        plt.close()