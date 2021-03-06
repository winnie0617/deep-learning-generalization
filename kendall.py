from __future__ import print_function
from ctypes import resize
from socketserver import DatagramRequestHandler
import models
import numpy
import complexity

def corr_fun(comp_measure_list, gen_gap_list):
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
                running_sum += numpy.sign(
                    comp_measure_list[i] - comp_measure_list[j]
                ) * numpy.sign(gen_gap_list[i] - gen_gap_list[j])
    return running_sum / cardinality


def basic_kendall(
    bs, lr, epochs, dp, comp_measure="VC", norm_measure="param_norm", lst="True"
):
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
    gen_gap_list = [
        train_loss_list[i] - test_loss_list[i] for i in range(len(train_loss_list))
    ]
    if comp_measure == "VC":
        vc_list = complexity.VC_dimension(model_list)
        return vc_list if lst else corr_fun(vc_list, gen_gap_list)
    else:
        norm_list = complexity.network_norm(model_list, norm_measure)
        return norm_list if lst else corr_fun(norm_list, gen_gap_list)


if __name__ == "__main__":
    import pandas as pd
    dataset = "MNIST"
    model_name = "conv"
    res = pd.read_csv(f"results/{dataset}-{model_name}.csv", index_col=0)
    measures = ["param_norm", "spectral_orig", "spec", "vc_dim"]
    hp_varied = ["depth", "width", "lr", "dropout"]
    kendall_dict = {}
    # Calculate correlation
    for measure in measures:
        corr = corr_fun(res[measure], res["l_test"]-res["l_train"])
        kendall_dict[measure] = [corr]
        print(f"{measure}: {corr}")
    res2 = pd.DataFrame(kendall_dict)
    res2.to_csv(f"results/{dataset}-{model_name}-all-correlation.csv")
