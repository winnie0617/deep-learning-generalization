from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from sklearn.model_selection import ParameterGrid

def make_nn(width, depth, dropout):
    ''' Return network in network model with given width, depth and dropout'''

    def nin_block(in_channels, out_channels, kernel_size, strides, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1), nn.ReLU(),
            nn.Dropout(dropout))
        
    def stack_blocks(depth, cin):
        net = nn.Sequential(
            nin_block(cin, width, kernel_size=3, strides=2, dropout=dropout),
            nin_block(width, width, kernel_size=3, strides=2, dropout=dropout))
        i = depth - 2
        while i > 0:
            net.add(nin_block(width, width, kernel_size=3, strides=2))
            net.add(nin_block(width, width, kernel_size=3, strides=2))
            i -= 2
        return net

    cin = 3 # TODO: change for MNIST
    net = nn.Sequential(
        stack_blocks(depth, cin),
        # There are 10 label classes
        nn.Conv2d(width, 10, 1, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        # Transform the four-dimensional output into two-dimensional output with a
        # shape of (batch size, 10)
        nn.Flatten())
    
    return net

import torch.nn as nn
import torch.nn.functional as F

model = make_nn(2*96, 2, 0.25)
num_epoch=1

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

train_losses=[]
valid_losses=[]

for epoch in range(1,num_epoch+1):
  train_loss=0.0
  valid_loss=0.0

  model.train()
  for img,lbl in train_ds_loader:
    img=img.cuda()
    lbl=lbl.cuda()

    optimizer.zero_grad()
    predict=model(img)
    loss=loss_fn(predict,lbl)
    loss.backward()
    optimizer.step()
    train_loss+=loss.item()*img.size(0)

  model.eval()
  for img,lbl in test_ds_loader:
    img=img.cuda()
    lbl=lbl.cuda()

    predict=model(img)
    loss=loss_fn(predict,lbl)

    valid_loss+=loss.item()*img.size(0)

  train_loss=train_loss/len(train_ds_loader.sampler) 
  valid_loss=valid_loss/len(test_ds_loader.sampler)

  train_losses.append(train_loss)
  valid_losses.append(valid_loss)

  print('Epoch:{} Train Loss:{:.4f} valid Losss:{:.4f}'.format(epoch,train_loss,valid_loss))    