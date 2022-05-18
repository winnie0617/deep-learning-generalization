
import nn

hp = {"batch_size":64, "lr":1}
model, train_loss, test_loss = nn.get_model(hp, epochs=1)
print('Train Loss: %.3f'%(train_loss))
print('Test Loss: %.3f'%(test_loss))
print(nn.get_weights(model))

# bs = [128]
# dp = [0.25, 0.5] 
# lr = [1]
# epochs = [1]
# depth = [1,2]
# width = [8,16]
# model_list, train_loss_list, test_loss_list = nn.get_models(bs, lr, epochs, dp)
# print('# of Models:' + str(len(model_list)))
# print(train_loss_list)
# print(test_loss_list)