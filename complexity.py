from __future__ import print_function
import models
import torch
import torch.nn as nn
from math import log10


def get_vc(model):
    """
    Returns VC dimension of a given model

    Tightest VC Dimension for a regular nn is O(WLlog(W)), where W = # of 
    weights, L = # of layers (depth)

    Parameters 
        model (Net): the neural network of interest
    """
    nn_weight_num = 0
    fc_weight_num = 0
    L = -1
    for layer in model.layer:
        if isinstance(layer, nn.Linear):
            # for linear activation, w_i = (input_i + 1)*output_i
            L += 1
            fc_weight_num += (layer.in_features + 1) * layer.out_features
        elif isinstance(layer, nn.Conv2d):
            # in general, # of w_i = (kernel_size^2 * output_{i-1} + 1)*
            # output_{i}, output_i = input_i-1m so 
            # total # of weights = \sum_{i} w_{i}
            L += 1
            nn_weight_num += (
                (layer.kernel_size[0] ** 2) * layer.in_channels + 1
            ) * layer.out_channels
    W = nn_weight_num + fc_weight_num
    return W * L * log10(L)

def VC_dimension(model_list):
    """
    Computes the VC dimension of each network model in model_list

    Tightest VC Dimension for a regular nn is O(WLlog(W)), 
    where W = # of weights, L = # of layers

    Parameters
        model_list (list[Net]): a list of neural network models generated 
            from the hyperparameter space
    """
    vc_list = []
    for model in model_list:
        vc_list.append(get_vc(model))
    return vc_list


def network_norm(model_list, norm_measure="param_norm"):
    """
    Returns the specified norm_measure for each network model in model_list

    There exists a mathematical formula for each norm_measure, so each one
    is calculated separately in accordance to its closed mathematical form 

    Parameters
        model_list (list[Net]): a list of neural network models generated from 
            the hyperparameter space
        norm_measure (str) the type of norm to use, which is one of 
            'param_norm', 'spectral_orig', 'path_norm', or 'spec'
    """
    def has_parameters(layer):
        try:
            temp = layer.weight
            return True
        except:
            return False
    norm_list = []
    if norm_measure == "param_norm": 
        for model in model_list:
            param_norm = 0
            for layer in model.layer:
                if has_parameters(layer):
                    param_norm += torch.norm(layer.weight, p="fro") ** 2
            norm_list.append(float(param_norm.detach().numpy()))
    elif norm_measure == "spectral_orig": 
        for model in model_list:
            first = 1
            second = 0
            for layer in model.layer:
                if has_parameters(layer):
                    two_norm = torch.norm(layer.weight, p=2) ** 2
                    first *= two_norm
                    second += (torch.norm(layer.weight, p="fro") ** 2) / two_norm
            norm_list.append(
                float(first.detach().numpy()) * float(second.detach().numpy())
            )
    elif norm_measure == "path_norm":  
        for model in model_list:
            path_norm = torch.sum(torch.square(models.get_weights(model)))
            norm_list.append(float(path_norm.detach().numpy()))
    elif norm_measure == "spec":  
        for model in model_list:
            depth = -1
            spec_norm = 1
            for layer in model.layer:
                if has_parameters(layer):
                    depth += 1
                    spec_norm = torch.norm(layer.weight, p=2) ** 2
            res = float(spec_norm.detach().numpy())
            norm_list.append(depth * (res ** (1 / depth)))
    return norm_list