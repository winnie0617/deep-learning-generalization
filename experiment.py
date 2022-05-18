import nn
import kendall_correlation

# hp = {"batch_size":64, "lr":1}
# model, train_loss, test_loss = nn.get_model(hp, epochs=1)
# print('Train Loss: %.3f'%(train_loss))
# print('Test Loss: %.3f'%(test_loss))
# print(nn.get_weights(model))

bs = [64, 128]
lr = [1]
epochs = [1]
dp = [0.25, 0.5]
depth = [1,2]
width = [8]
# model_list, train_loss_list, test_loss_list = nn.get_models(bs, lr, epochs, dp)
vc_list = kendall_correlation.basic_kendall(bs, lr, epochs, dp, measure='VC', lst=True)
# VC_ken_corr = kendall_correlation.basic_kendall(bs, lr, epochs, dp, measure='VC', lst=False) -- NaN atm cuz there's no change in width/depth

# print('# of Models:' + str(len(model_list)))
# print('Training Loss:' + str(train_loss_list))
# print('Testing Loss:' +str(test_loss_list))
print('VC_list:' + str(vc_list))
# print('correlation' + str(VC_ken_corr))