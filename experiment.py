import models
import kendall
import complexity
import pandas as pd

### This file contains the code to make the results folder

def make_hp_list(batch_size_lst, depth_lst, width_lst, lr_lst, epochs_lst, 
                dropout_lst):
    """
    Returns a dictionary of specified parameter list, called a hyperparameter
    space.  
    
    Each possible permutation of the hyperparameter space in a unique 
    model architecture that will be trained.

    Parameters
        batch_size_lst (int[list]): a list of batch sizes to be trained
        depth_lst (int[list]): a list of depths to be trained
        width_lst (int[list]): a list of widths to be trained
        lr_lst (int[list]): a list of learning rates to be trained
        epochs_lst (int[list]): a list of epochs to be trained
        dropout_lst (int[list]): a list of dropout probabilities to be trained
    """
    hp_lst = {
        "batch_size": batch_size_lst, 
        "depth": depth_lst,
        "width": width_lst,
        "lr": lr_lst,
        "epochs": epochs_lst,
        "dropout": dropout_lst
    }
    return hp_lst

def model_results(hp_list, dataset, model_name):
    """
    Return (1): the training and testing results as a dataframe 
           (2): the list of architectural models used 
    for each permutable model in the hyperparameter space defined by hp_list 
    under the specified architectural choice. 

    Parameters
        hp_list (dict): a dictionary of specificed parameter list
        dataset (str): either "CIFAR10" or "MNIST", which are the datasets
            chosen for this research
        model_name (str): either "NiN" or "conv", which are the major 
            architectures adopted in this research 
    """
    grid, model_list, train_loss_list, test_loss_list = models.get_models(
    hp_list, dataset, model_name)
    res = pd.DataFrame(grid)
    res["l_train"] = train_loss_list
    res["l_test"] = test_loss_list
    return res, model_list

def result(res, model_name, model_lst):
    """
    Returns complexities result dataframe for each corresponding model 
    and an overall kendall correlation dataframe.
    
    Also generate the corresponding dataframe results in the results folder as 
    a .csv file.

    Parameters
        res (df): the dataframe containing training & testing results for 
                  each model
        model_name (str): either "NiN" or "conv", which are the major 
            architectures adopted in this research 
        model_lst (str[lst]): the list of architectural models used
    """
    measures = ["param_norm", "spectral_orig", "spec", "vc_dim"]
    for norm_type in measures:   
        res[norm_type] = complexity.network_norm(model_lst, norm_type)
    res["vc_dim"] = complexity.VC_dimension(model_lst)
    res.to_csv(f"results/{dataset}-{model_name}.csv")

    kendall_dict = {}
    for measure in measures:
        corr = kendall.corr_fun(res[measure], res["l_test"]-res["l_train"])
        kendall_dict[measure] = [corr]
    res2 = pd.DataFrame(kendall_dict)
    res2.to_csv(f"results/{dataset}-{model_name}-correlation.csv")
    return res, res2

# The parameters to generate the files in the results folder 
hp_list = make_hp_list([32], [2], [32,64,96], [0.5,0.75,1], [11], 
                       [0.25,0.5,0.75])
dataset, model_name, dataset = "CIFAR10", "NiN", "CIFAR10"
# dataset, model_name, dataset = "MNIST", "conv", "MNIST"
res, model_list = model_results(hp_list, dataset, model_name)
comp_res, kendall_res = result(res, model_name, model_list)

