from __future__ import print_function
import nn
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    kendall, _ = kendalltau(comp_measure_list, gen_gap_list)
    return kendall

#to do
def basic_kendall(w, bs, lr, epochs, dp, d, op):
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
    """
    model_list, train_loss_list, test_loss_list = nn.get_models(bs, lr, epochs, dp)
    gen_gap_list = [train-test for train, test in zip(train_loss_list, test_loss_list)]
    # todo: generate comp_measure_list from a list of models. 

#to do
def VC_dimension():
    """
    Computes the VC dimension of a neural network 
    """

#to do
def network_norm():
    """
    Computes the spectral norm of a neural network
    """