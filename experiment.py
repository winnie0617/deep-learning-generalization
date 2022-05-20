import models
import kendall_correlation

# hp = {"batch_size":64, "lr":1}
# model, train_loss, test_loss = models.get_model(hp, epochs=1, depth=2)
# print('Train Loss: %.3f'%(train_loss))
# print('Test Loss: %.3f'%(test_loss))
# print(models.get_weights(model))

hp_list = {"batch_size": [64, 128], "depth": [2,4], "width": [2*96], "lr": [1], "epochs": [1], "dropout": [0.25,0.5]}
dataset = "CIFAR10"
dataset = "MNIST"

def test_norm_kendall():
    # norm_list = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='spectral_orig', lst=True)
    # norm_list2 = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='path_norm', lst=True)
    # norm_list3 = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='spec', lst=True)
    norm_ken_corra = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='param_norm', lst=True)
    norm_ken_corr = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='norm', norm_measure='param_norm', lst=False)
    # print('norm list:' + str(norm_list))
    # print('norm list2:' + str(norm_list2))
    # print('norm list3:' + str(norm_list3))
    print('norm list3:' + str(norm_ken_corra))
    print('norm kendall: %.3f' % norm_ken_corr)

def test_VC_kendall():
    vc_list = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='VC', lst=True)
    VC_ken_corr = kendall_correlation.basic_kendall(bs, lr, epochs, dp, comp_measure='VC', lst=False)
    print('VClist:' + str(vc_list))
    print('VC kendall: %.3f' % VC_ken_corr)

def test_network():
    model_list, train_loss_list, test_loss_list = models.get_models(hp_list, dataset)
    print('# of Models: ' + str(len(model_list)))
    print('Training Loss:' + str(train_loss_list))
    print('Testing Loss:' +str(test_loss_list))
    
# test_norm_kendall()
# test_VC_kendall()
test_network()
