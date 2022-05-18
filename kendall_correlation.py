from __future__ import print_function
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
    for i,j in zip(n,n):
        if i != j:
            running_sum += numpy.sign(comp_measure_list[i] - comp_measure_list[j])*numpy.sign(gen_gap_list[i] - gen_gap_list[j])
    return running_sum/n

def basic_kendall(bs, lr, epochs, dp, measure='VC', lst='True'):
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
    gen_gap_list = [train-test for train, test in zip(train_loss_list, test_loss_list)]
    if measure == 'VC':
        vc_list = VC_dimension(model_list)
        return vc_list if lst else correlation_function(vc_list, gen_gap_list)
    else:
        return correlation_function(network_norm(model_list), gen_gap_list)
    

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
def network_norm(model_list):
    """
    Computes the spectral norm of each network model in model_list
    """
    norm_list = []
    for model in model_list:
        norm_list.append(nn.get_weights(model))
    return norm_list