import models
import kendall
import complexity
import pandas as pd

# hp = {"batch_size":64, "lr":1}
# model, train_loss, test_loss = models.get_model(hp, epochs=1, depth=2)
# print('Train Loss: %.3f'%(train_loss))
# print('Test Loss: %.3f'%(test_loss))
# print(models.get_weights(model))

# hp_list = {
#     "batch_size": [32],
#     "depth": [2], # min: 2
#     "width": [3*96, 2*96, 3*96], 
#     "lr": [0.5, 1],
#     "epochs": [6],
#     "dropout": [0.25, 0.5],
# }

hp_list = {
    "batch_size": [64],
    "depth": [2, 4, 6], # min: 2
    "width": [32, 64, 96], 
    "lr": [0.1, 1],
    "epochs": [3],
    "dropout": [0.25, 0.5],
}
dataset = "MNIST"
# dataset = "MNIST"
# model_name = "NiN"
model_name = "conv"


# def test_norm_kendall():
#     # norm_list = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='spectral_orig', lst=True)
#     # norm_list2 = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='path_norm', lst=True)
#     # norm_list3 = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='spec', lst=True)
#     norm_ken_corra = kendall.basic_kendall(
#         bs, lr, epochs, dp, comp_measure="norm", norm_measure="param_norm", lst=True
#     )
#     norm_ken_corr = kendall.basic_kendall(
#         bs, lr, epochs, dp, comp_measure="norm", norm_measure="param_norm", lst=False
#     )
#     # print('norm list:' + str(norm_list))
#     # print('norm list2:' + str(norm_list2))
#     # print('norm list3:' + str(norm_list3))
#     print("norm list3:" + str(norm_ken_corra))
#     print("norm kendall: %.3f" % norm_ken_corr)


# def test_VC_kendall():
#     vc_list = kendall.basic_kendall(bs, lr, epochs, dp, comp_measure="VC", lst=True)
#     VC_ken_corr = kendall.basic_kendall(
#         bs, lr, epochs, dp, comp_measure="VC", lst=False
#     )
#     print("VClist:" + str(vc_list))
#     print("VC kendall: %.3f" % VC_ken_corr)


# def test_network():
#     grid, model_list, train_loss_list, test_loss_list = models.get_models(
#         hp_list, dataset
#     )
#     print("# of Models: " + str(len(model_list)))
#     print("Grid " + str(grid))
#     print("Training Loss:" + str(train_loss_list))
#     print("Testing Loss:" + str(test_loss_list))


# test_norm_kendall()
# test_VC_kendall()
# test_network()

# Store hp, complexity measures, and test loss in pandas dataframe
grid, model_list, train_loss_list, test_loss_list = models.get_models(hp_list, dataset, model_name)
res = pd.DataFrame(grid)
res["l_train"] = train_loss_list
res["l_test"] = test_loss_list

# Complexity measures
measures = ["param_norm", "spectral_orig", "spec", "vc_dim"]
 
for norm_type in ["param_norm", "spectral_orig", "spec"]:  # TODO: skipping path norm
    res[norm_type] = complexity.network_norm(model_list, norm_type)
res["vc_dim"] = complexity.VC_dimension(model_list)

res.to_csv(f"results/{dataset}-{model_name}.csv")

kendall_dict = {}
# Calculate correlation
for measure in measures:
    corr = kendall.corr_fun(res[measure], res["l_test"]-res["l_train"])
    kendall_dict[measure] = [corr]
    print(f"{measure}: {corr}")
res2 = pd.DataFrame(kendall_dict)
res2.to_csv(f"results/{dataset}-{model_name}-correlation.csv")