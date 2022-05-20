from __future__ import print_function
from ctypes import resize
from socketserver import DatagramRequestHandler
import models
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
    model_list, train_loss_list, test_loss_list = models.get_models(bs, lr, epochs, dp)
    gen_gap_list = [train_loss_list[i]-test_loss_list[i] for i in range(len(train_loss_list))]
    if comp_measure == 'VC':
        vc_list = VC_dimension(model_list)
        return vc_list if lst else correlation_function(vc_list, gen_gap_list)
    else:
        norm_list = network_norm(model_list, norm_measure)
        return norm_list if lst else correlation_function(norm_list, gen_gap_list)
    



