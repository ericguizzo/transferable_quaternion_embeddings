import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ATLoss():
    def __init__(self, pretrained_model, verbose=False):
        '''
        Compute anti-transfer loss term between a pre-trained feature extractor
        and an identical model being trained. The loss reflect the diversity
        of the developed deep features at a specific layer (at_layer).
        '''
        self.pretrained_model = pretrained_model
        self.cos_similarity = nn.CosineSimilarity(dim=-1)
        self.epsylon = 1e-10
        self.verbose = verbose

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

    def pairwise_aggregation(self, a, b):
        a = a.view(a.shape[0], a.shape[1], a.shape[2]*a.shape[3])
        b = b.view(b.shape[0], b.shape[1], b.shape[2]*b.shape[3])
        ch = a.shape[1]

        a = a.repeat(1, 1, ch)
        a = a.view(a.shape[0], 1, a.shape[1]*a.shape[2])

        b = b.view(b.shape[0], 1, b.shape[1]*b.shape[2])
        b = b.repeat(1, 1, ch)

        return a, b

    def stochastic_pairwise_aggregation(self, a, b):
        b_dim, c_dim, x_dim, y_dim = a.shape

        if b_dim > 1:
            q = int(c_dim / b_dim)  #chans/batch so that 1 epoch contains all channels
            begin = np.random.randint(c_dim - q)
            end = begin + q
        else:
            begin = np.random.randint(c_dim)
            end = begin + 1

        a = a[:,begin:end,:,:]
        b = b[:,begin:end,:,:]

        a = a.view(a.shape[0], a.shape[1], a.shape[2]*a.shape[3])
        b = b.view(b.shape[0], b.shape[1], b.shape[2]*b.shape[3])
        ch = a.shape[1]

        a = a.repeat(1, 1, ch)
        a = a.view(a.shape[0], 1, a.shape[1]*a.shape[2])

        b = b.view(b.shape[0], 1, b.shape[1]*b.shape[2])
        b = b.repeat(1, 1, ch)

        return a, b

    def loss(self, input, current_model, beta=1. ,
                    aggregation='gram', distance='cos_squared'):
        '''
        - input = input tensor
        - current_model = model being trained
        - at_layer = compute AT loss in this layer
        - beta = at_loss weight
        - aggregation = function to aggregate channels
        - distance = distance function between the aggregated feature maps
        '''
        #extract features until at_layer
        pre_feat, _, _, _, _ = self.pretrained_model(input)
        curr_feat, _, _, _, _ = current_model(input)

        if aggregation == 'none':
            #no aggregation (channel permutation makes AT useless)
            pass

        elif aggregation == 'mean':
            #channel-wise mean
            curr_feat = curr_feat.mean(1).view(curr_feat.shape[0], 1,
                        curr_feat.shape[2], curr_feat.shape[3])
            pre_feat = pre_feat.mean(1).view(pre_feat.shape[0], 1,
                        pre_feat.shape[2], pre_feat.shape[3])

        elif aggregation == 'sum':
            #channel-wise sum
            curr_feat = curr_feat.sum(1).view(curr_feat.shape[0], 1,
                        curr_feat.shape[2], curr_feat.shape[3])
            pre_feat = pre_feat.sum(1).view(pre_feat.shape[0], 1,
                        pre_feat.shape[2], pre_feat.shape[3])

        elif aggregation == 'mul_comp':
            #channel-wise multiplication
            #elevated to the power of 0.001 to not obtain very small values
            curr_feat = curr_feat + self.epsylon
            pre_feat = pre_feat + self.epsylon
            curr_feat = curr_feat ** 0.001
            pre_feat = pre_feat ** 0.001
            curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                        curr_feat.shape[2], curr_feat.shape[3])
            pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                        pre_feat.shape[2], pre_feat.shape[3])

        elif aggregation == 'max':
            #channel-wise max function
            curr_feat = F.max_pool3d(curr_feat, kernel_size=[curr_feat.shape[1],1,1])
            pre_feat = F.max_pool3d(pre_feat, kernel_size=[pre_feat.shape[1],1,1])

        elif aggregation == 'gram':
            #gramian matrix channel aggregation
            curr_feat = self.gram_matrix(curr_feat) / 0.001
            pre_feat = self.gram_matrix(pre_feat) / 0.001

        elif aggregation == 'pairwise':
            #create all possible channel combinations in each data point (highly memory-demanding)
            curr_feat, pre_feat = self.pairwise_aggregation(curr_feat, pre_feat)

        elif aggregation == 'stochastic_pairwise':
            #reduced aggregation: all combinations appear in one batch
            #less memory demanding
            curr_feat, pre_feat = self.stochastic_pairwise_aggregation(curr_feat, pre_feat)

        else:
            raise NotImplementedError('Non supported aggregation function selected')

        if distance == 'mse_sigmoid':
            loss = F.mse_loss(curr_feat, pre_feat)
            loss = F.sigmoid(loss)
            loss = loss * -1

        elif distance == 'cos_squared':
            if len(curr_feat.shape) > 3:
                curr_feat = curr_feat.view(curr_feat.shape[0], curr_feat.shape[1], curr_feat.shape[2]*curr_feat.shape[3])
                pre_feat = pre_feat.view(pre_feat.shape[0], pre_feat.shape[1], pre_feat.shape[2]*pre_feat.shape[3])
            loss = self.cos_similarity(curr_feat, pre_feat)
            loss = torch.abs(loss)
            loss = torch.mean(loss)
            loss = loss ** 2
            loss = loss * beta

        else:
            raise NotImplementedError('Non-supported distance function selected')

        #Weight multiplication
        loss = loss * beta

        if self.verbose:
            #monitor features sum during training as the model may learn to just increase the
            #features magnitude
            print ('Features sum: ', torch.sum(curr_feat).item(), torch.sum(pre_feat).item())

        return loss
