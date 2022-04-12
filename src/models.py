import torch
from torch import nn
import torch.nn.functional as F
from quaternion_layers import (QuaternionConv, QuaternionLinear,
                               QuaternionTransposeConv)
from qbn import QuaternionBatchNorm2d
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor

VGG_types = {
    "simple": [16,"M",32,"M",128,],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",],
    "VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M",],
    "VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M",]
    }


class VGGNet(nn.Module):
    def __init__(self,
                architecture='VGG16',
                classifier_dropout=0.5,
                flatten_dim=32768,
                verbose=True,
                quat=False,
                num_classes = 4
                ):
        super(VGGNet, self).__init__()
        self.quat = quat
        if quat:
            self.in_channels = 4
        else:
            self.in_channels = 1
        self.verbose = verbose
        self.flatten_dim = flatten_dim
        self.last_dim = [i for i in VGG_types[architecture] if type(i) != str][-1]
        self.first_dim = [i for i in VGG_types[architecture] if type(i) != str][0]

        self.features = self.create_conv_layers(VGG_types[architecture])

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)
                             ]
        classifier_layers_q = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)
                             ]
        if quat:
            self.classifier = nn.Sequential(*classifier_layers_q)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                if self.quat:
                    c = QuaternionConv(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1))
                else:
                    c = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1))
                layers += [c,
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print ('input: ', x.shape)
        x = self.features(x)
        if self.verbose:
            print ('features: ', x.shape)
        x = torch.flatten(x, start_dim=1)
        if self.verbose:
            print('flatten: ', x.shape)
        x = self.classifier(x)
        if self.verbose:
            print('classification: ', x.shape)
        return x

class AlexNet(nn.Module):

    def __init__(self, quat=False, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        if quat:
            self.features = nn.Sequential(
                QuaternionConv(4, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                QuaternionConv(64, 192, kernel_size=5, padding=2, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                QuaternionConv(192, 384, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                QuaternionConv(384, 256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                QuaternionConv(256, 256, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                QuaternionLinear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                QuaternionLinear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class simple_autoencoder(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=2048 ,flatten_dim=16384,
                 classifier_dropout=0.5, embeddings_dim=[64,64], num_classes=5):
        super(simple_autoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        #self.conv6 = nn.Conv2d(256, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.tconv2_bn = QuaternionBatchNorm2d(16)
        #self.hidden = nn.Linear(flatten_dim, hidden_size*4)
        #self.decoder_input = nn.Linear(hidden_size*4, flatten_dim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        if quat:
            #self.t_conv0 = QuaternionTransposeConv(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv1 = QuaternionTransposeConv(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv2 = QuaternionTransposeConv(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv3 = QuaternionTransposeConv(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv4 = QuaternionTransposeConv(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv5 = QuaternionTransposeConv(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            #self.t_conv0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1,output_padding=1)
            self.t_conv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1,output_padding=1)
            self.t_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,output_padding=1)
            self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1)
            self.t_conv4 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,output_padding=1)
            self.t_conv5 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1,output_padding=1)

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]
        classifier_layers_quat = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        #self.classifier_valence = nn.Sequential(*classifier_layers)
        #self.classifier_arousal = nn.Sequential(*classifier_layers)
        #self.classifier_dominance = nn.Sequential(*classifier_layers)

        if classifier_quat:
            self.classifier = nn.Sequential(*classifier_layers_quat)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv2_bn(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        #x = F.relu(self.conv6(x))
        #x = self.pool(x)


        #print ('CAZZOOOOOOOOOO', x.shape)
        #hidden dim
        x = torch.flatten(x, start_dim=1)
        #x = torch.sigmoid(self.hidden(x))
        #print (x.shape)

        return x

    def decode(self, x):
        #x = F.relu(self.decoder_input(x))

        x = x.view(-1, 256, 16, 4)
        #x1 = F.relu(self.t_conv0(x1))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))

        x = F.relu(self.t_conv4(x))
        x = self.tconv2_bn(x)
        x = torch.sigmoid(self.t_conv5(x))

        return x


    def get_embeddings(self, x):
        x = self.encode(x)
        x = x.view(-1, 4, self.embeddings_dim[0], self.embeddings_dim[1])
        return x, 'dummy'


    def forward(self, x):
        #a = self.get_embeddings(x)
        x = self.encode(x)
        pred = self.classifier(x)
        x = self.decode(x)

        return x, pred


class simple_autoencoder_2(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=4096 ,flatten_dim=16384,
                 classifier_dropout=0.5, embeddings_dim=[64,64], num_classes=5):
        super(simple_autoencoder_2, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(2)
        self.conv2_bn = nn.BatchNorm2d(4)
        #self.conv3_bn = nn.BatchNorm2d(4)

        #self.hidden = nn.Linear(flatten_dim, hidden_size*4)
        #self.decoder_input = nn.Linear(hidden_size*4, flatten_dim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        if quat:
            self.t_conv1 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.tconv1_bn = QuaternionBatchNorm2d(4)
            self.tconv2_bn = QuaternionBatchNorm2d(4)
        else:
            self.t_conv1 = nn.ConvTranspose2d(4, 4, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = nn.ConvTranspose2d(4, 2, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=1)
            self.tconv1_bn = nn.BatchNorm2d(4)
            self.tconv2_bn = nn.BatchNorm2d(2)

        classifier_layers = [nn.Linear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        classifier_layers_quat = [QuaternionLinear(flatten_dim, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             QuaternionLinear(4096, 4096),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(4096, num_classes)]

        if classifier_quat:
            self.classifier = nn.Sequential(*classifier_layers_quat)
        else:
            self.classifier = nn.Sequential(*classifier_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=[2,2])
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=[2,1])
        x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=[2,1])

        return x


    def decode(self, x):
        x = F.relu(self.t_conv1(x))
        x = self.tconv1_bn(x)
        x = F.relu(self.t_conv2(x))
        x = self.tconv2_bn(x)
        x = torch.sigmoid(self.t_conv3(x))

        return x


    def get_embeddings(self, x):
        x = self.encode(x)

        return x, 'dummy'

    def forward(self, x):
        x = self.encode(x)
        x_pred = torch.flatten(x, start_dim=1)
        pred = self.classifier(x_pred)
        x = self.decode(x)

        return x, pred


class simple_autoencoder_2_vad(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=4096 ,flatten_dim=16384,
                 classifier_dropout=0.5, embeddings_dim=[64,64], num_classes=5, ):
        super(simple_autoencoder_2_vad, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(2)
        self.conv2_bn = nn.BatchNorm2d(4)
        #self.conv3_bn = nn.BatchNorm2d(4)

        #self.hidden = nn.Linear(flatten_dim, hidden_size*4)
        #self.decoder_input = nn.Linear(hidden_size*4, flatten_dim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        if quat:
            self.t_conv1 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = QuaternionTransposeConv(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.tconv1_bn = QuaternionBatchNorm2d(4)
            self.tconv2_bn = QuaternionBatchNorm2d(4)
        else:
            self.t_conv1 = nn.ConvTranspose2d(4, 4, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = nn.ConvTranspose2d(4, 2, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=1)
            self.tconv1_bn = nn.BatchNorm2d(4)
            self.tconv2_bn = nn.BatchNorm2d(2)

        classifier_layers_discrete = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, num_classes)]

        classifier_layers_valence = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        classifier_layers_arousal = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        classifier_layers_dominance = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        self.classifier_discrete = nn.Sequential(*classifier_layers_discrete)
        self.classifier_valence = nn.Sequential(*classifier_layers_valence)
        self.classifier_arousal = nn.Sequential(*classifier_layers_arousal)
        self.classifier_dominance = nn.Sequential(*classifier_layers_dominance)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=[2,2])
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=[2,1])
        x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=[2,1])

        return x


    def decode(self, x):
        x = F.relu(self.t_conv1(x))
        x = self.tconv1_bn(x)
        x = F.relu(self.t_conv2(x))
        x = self.tconv2_bn(x)
        x = torch.sigmoid(self.t_conv3(x))

        return x


    def get_embeddings(self, x):
        x = self.encode(x)
        _ = "dummy"

        return x, _, _, _, _


    def forward(self, x):

        x = self.encode(x)

        x_discrete = torch.flatten(x[:,0,:,:], start_dim=1)
        x_valence = torch.flatten(x[:,1,:,:], start_dim=1)
        x_arousal = torch.flatten(x[:,2,:,:], start_dim=1)
        x_dominance = torch.flatten(x[:,3,:,:], start_dim=1)

        pred_discrete = self.classifier_discrete(x_discrete)
        pred_valence = torch.sigmoid(self.classifier_valence(x_valence))
        pred_arousal = torch.sigmoid(self.classifier_arousal(x_arousal))
        pred_dominance = torch.sigmoid(self.classifier_dominance(x_dominance))

        x = self.decode(x)

        return x, pred_discrete, pred_valence, pred_arousal, pred_dominance


class simple_autoencoder_2_vad_mod(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=4096 ,flatten_dim=16384,
                 classifier_dropout=0.5, embeddings_dim=[64,64], num_classes=5, batchnorm=False):
        super(simple_autoencoder_2_vad_mod, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 4, 3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(2)
        self.conv2_bn = nn.BatchNorm2d(4)
        #self.conv3_bn = nn.BatchNorm2d(4)
        self.batchnorm = batchnorm

        #self.hidden = nn.Linear(flatten_dim, hidden_size*4)
        #self.decoder_input = nn.Linear(hidden_size*4, flatten_dim)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        if quat:
            self.t_conv1 = QuaternionTransposeConv(4, 128, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = QuaternionTransposeConv(128, 64, kernel_size=3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = QuaternionTransposeConv(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.t_conv4 = QuaternionTransposeConv(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0)
            self.t_conv5 = QuaternionTransposeConv(16, 4, kernel_size=3, stride=1, padding=1, output_padding=0)

            self.tconv1_bn = QuaternionBatchNorm2d(4)
            self.tconv2_bn = QuaternionBatchNorm2d(4)
        else:
            self.t_conv1 = nn.ConvTranspose2d(4, 64, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv2 = nn.ConvTranspose2d(64, 32, 3, stride=[2,1], padding=1, output_padding=[1,0])
            self.t_conv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
            self.t_conv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=0)

            self.tconv1_bn = nn.BatchNorm2d(4)
            self.tconv2_bn = nn.BatchNorm2d(2)

        classifier_layers_discrete = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, num_classes)]

        classifier_layers_valence = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        classifier_layers_arousal = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        classifier_layers_dominance = [nn.Linear(flatten_dim//4, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, self.hidden_size),
                             nn.ReLU(),
                             nn.Dropout(p=classifier_dropout),
                             nn.Linear(self.hidden_size, 1)]

        self.classifier_discrete = nn.Sequential(*classifier_layers_discrete)
        self.classifier_valence = nn.Sequential(*classifier_layers_valence)
        self.classifier_arousal = nn.Sequential(*classifier_layers_arousal)
        self.classifier_dominance = nn.Sequential(*classifier_layers_dominance)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=[2,2])
        if self.batchnorm:
            x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=[2,1])
        if self.batchnorm:
            x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=[2,1])
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x


    def decode(self, x):
        x = F.relu(self.t_conv1(x))
        if self.batchnorm:
            x = self.tconv1_bn(x)
        x = F.relu(self.t_conv2(x))
        if self.batchnorm:
            x = self.tconv2_bn(x)
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = torch.sigmoid(self.t_conv5(x))

        return x


    def get_embeddings(self, x):
        x = self.encode(x)
        _ = "dummy"

        return x, _, _, _, _


    def forward(self, x):

        x = self.encode(x)

        x_discrete = torch.flatten(x[:,0,:,:], start_dim=1)
        x_valence = torch.flatten(x[:,1,:,:], start_dim=1)
        x_arousal = torch.flatten(x[:,2,:,:], start_dim=1)
        x_dominance = torch.flatten(x[:,3,:,:], start_dim=1)

        pred_discrete = self.classifier_discrete(x_discrete)
        pred_valence = torch.sigmoid(self.classifier_valence(x_valence))
        pred_arousal = torch.sigmoid(self.classifier_arousal(x_arousal))
        pred_dominance = torch.sigmoid(self.classifier_dominance(x_dominance))

        x = self.decode(x)

        return x, pred_discrete, pred_valence, pred_arousal, pred_dominance



#__all__ = ['ResNet','resnet50']

class dual_simple_autoencoder(nn.Module):
    def __init__(self, quat=True, classifier_quat=True, hidden_size=2048 ,flatten_dim=16384,
                 classifier_dropout=0.5, embeddings_dim=[64,64], num_classes=5):
        super(dual_simple_autoencoder, self).__init__()
        self.model_1 = simple_autoencoder(quat=quat, classifier_quat=classifier_quat, hidden_size=hidden_size ,flatten_dim=flatten_dim,
                     classifier_dropout=classifier_dropout, embeddings_dim=embeddings_dim, num_classes=num_classes)

        self.model_2 = simple_autoencoder(quat=quat, classifier_quat=classifier_quat, hidden_size=hidden_size ,flatten_dim=flatten_dim,
                    classifier_dropout=classifier_dropout, embeddings_dim=embeddings_dim, num_classes=num_classes)

    def get_embeddings(self, x):
        x1, _ = self.model_1.get_embeddings(x)
        x2, _ = self.model_2.get_embeddings(x)
        print (x1.shape, x2.shape)
        out = torch.cat((x1,x2), -2)
        print(out.shape)

        return out

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, quat: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""

    if quat:
        return QuaternionConv(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilatation=dilation)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, quat: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    if quat:
        return QuaternionConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        quat: bool = False,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            if quat:
                norm_layer = QuaternionBatchNorm2d
            else:
                norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, quat=quat)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quat=quat)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        quat: bool = False,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            if quat:
                norm_layer = QuaternionBatchNorm2d
            else:
                norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, quat=quat)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, quat=quat)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, quat=quat)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        quat: bool = False,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            if quat:
                norm_layer = QuaternionBatchNorm2d
            else:
                norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if quat:
            self.conv1 = QuaternionConv(4, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)


        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], quat=quat)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], quat=quat)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], quat=quat)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], quat=quat)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, quat: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, quat=quat),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, quat=quat))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, quat=quat))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
