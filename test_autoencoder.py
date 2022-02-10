import torch
from torch import nn
import numpy as np
from models import *
from torchsummary import summary
from torchvision import models
from torchsummary import summary
from torch import optim

x = torch.rand(1, 1, 512, 128)
'''
model = resnet50(num_classes=7, quat=True)
#print (model)
#x = model(x)
#print (x.shape)
#model_params = sum([np.prod(p.size()) for p in model.parameters()])
#print ('Total paramters: ' + str(model_params))

summary(model, (4,64,64))
'''

model = simple_autoencoder_2_vad(quat=True)
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

'''
num_optim_layers = 10

#A
c = 1
for p in model.parameters():
    if c <= 10:
        p.requires_grad = True
    else:
        p.requires_grad = False
    c += 1

optimizer = optim.Adam(list(model_emo.parameters()) +  list(model.parameters()))

#B
c = 1
r2Hemo_params = []
for p in model.parameters():
    p.requires_grad = True
    r2Hemo_params.append(p)
    c += 1
optimizer = optim.Adam(list(model_emo.parameters()) +  r2Hemo_params)




print ('input_dim', x.shape)
x, pred = model(x)
print ('output_dim', x.shape)
y = model(x)
print ('enc dim', y.shape)

#compute number of parameters
#model_params = sum([np.prod(p.size()) for p in model.parameters()])
#print ('Total paramters: ' + str(model_params))


model = r2he(verbose=True,
             latent_dim=100,
             quat=True,
             #flattened_dim=524288,
             architecture='VGG16')
print (model)

print ('TESTING DIMENSIONS')
print ('input_dim', x.shape)
x,v,a,d=model(x)

'''
