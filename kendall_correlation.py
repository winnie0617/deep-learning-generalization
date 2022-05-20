from __future__ import print_function
from ctypes import resize
from socketserver import DatagramRequestHandler
import nn as nnn
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from scipy.stats import kendalltau

def correlation_function(comp_measure_list, gen_gap_list):
    """
    Given list of complexity mesaures and associated generalization gaps, compute τ 

    comp_measure_list: list of μ(θ) for each hyperparameter set θ
    gen_gap_list: list of  g(θ) for each hyperparameter set θ 
    """
    n = len(comp_measure_list)
    cardinality = n**2 - n
    running_sum = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                running_sum += numpy.sign(comp_measure_list[i] - comp_measure_list[j])*numpy.sign(gen_gap_list[i] - gen_gap_list[j])
    return running_sum/cardinality

def basic_kendall(bs, lr, epochs, dp, comp_measure='VC', norm_measure='param_norm', lst='True'):
    """
    Computes the vanilla kendall correlation on a hyperparameter space choices Θ_1 x ... Θ_n. 

    Very space & time inefficient. 

    bs: list of batch size choices v
    lr: list of learning rate choices v
    epochs: list of epoch choices v
    dp: list of dropout probability choices
    w: list of network layer widths  
    d: list of network depth choices
    op: list of optimizer choices
    measure: either 'VC' or 'norm' (complexity measure to use)
    lst: boolean indicating whether to return a list of VC dimensions or the kendall correlation number
    """
    model_list, train_loss_list, test_loss_list = nnn.get_models(bs, lr, epochs, dp)
    gen_gap_list = [train_loss_list[i]-test_loss_list[i] for i in range(len(train_loss_list))]
    print('gen gap list :' + str(gen_gap_list))
    if comp_measure == 'VC':
        vc_list = VC_dimension(model_list)
        return vc_list if lst else correlation_function(vc_list, gen_gap_list)
    else:
        norm_list = network_norm(model_list, norm_measure)
        return norm_list if lst else correlation_function(norm_list, gen_gap_list)
    

def VC_dimension(model_list):
    """
    Computes the VC dimension of each network model in model_list

    Tightest VC Dimension for a regular nn is O(WLlog(W)), where W = # of weights, L = # of layers

    model_list: a list of neural network models generated from hyperparameter space
    """
    vc_list = []
    for model in model_list:
        vc_list.append(nnn.get_vc(model))
    return vc_list

#probably to do; kinda a stub rn
def network_norm(model_list, norm_measure ='param_norm'):
    """
    Computes the spectral norm of each network model in model_list

    model_list: a list of neural network models generated from hyperparameter space
    norm_measure : the type of norm to use ('param_norm', 'spectral_orig', 'path_norm', 'spec')
    """
    #each norm measure is so different that I had to just do them separately to avoid confusion.

    norm_list = []
    
    if norm_measure == 'param_norm': # formula 42
        for model in model_list:
            param_norm = 0
            for layer in model.children():
                if not isinstance(layer, nn.Dropout):
                    param_norm += torch.norm(layer.weight, p='fro')**2
            norm_list.append(float(param_norm.detach().numpy()))
    elif norm_measure == 'spectral_orig': # formula 28
        for model in model_list:
            first = 1
            second = 0
            for layer in model.children():
                if not isinstance(layer, nn.Dropout):
                    two_norm = torch.norm(layer.weight, p=2)**2
                    first *= two_norm
                    second += (torch.norm(layer.weight, p='fro')**2)/two_norm
            norm_list.append(float(first.detach().numpy())*float(second.detach().numpy()))
    elif norm_measure == 'path_norm': # formula 44
        for model in model_list:
            path_norm = torch.sum(torch.square(nnn.get_weights(model)))
            norm_list.append(float(path_norm.detach().numpy()))
    elif norm_measure == 'spec': # formula 35
        for model in model_list:
            depth = -1
            spec_norm = 1
            for layer in model.children():
                if not isinstance(layer, nn.Dropout):
                    depth+=1
                    spec_norm = torch.norm(layer.weight, p=2)**2
            res = float(spec_norm.detach().numpy())
            norm_list.append(depth*(res**(1/depth)))
    return norm_list


