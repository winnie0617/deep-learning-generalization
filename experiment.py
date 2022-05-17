
import nn
hp = {"batch_size":64, "lr":1}
model, test_loss = nn.get_model(hp, epochs=1)
print(test_loss)