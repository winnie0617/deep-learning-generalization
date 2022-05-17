from __future__ import print_function
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
    Given generalization gaps, compute τ 

    comp_measure_list: list of μ(θ) for each hyperparameter set θ
    gen_gap_list: list of  g(θ) for each hyperparameter set θ 
    """
    corr, _ = kendalltau(comp_measure_list, gen_gap_list)
    return corr

#to do
def basic_kendall():
    """
    Very space & time inefficient. 
    """
    ranking_list = []

    return

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