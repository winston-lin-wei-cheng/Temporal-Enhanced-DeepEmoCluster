#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import torch.nn as nn
import math


__all__ = [ 'VGG', 'vgg16']

class VGG(nn.Module):

    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.cnn_features = features
        self.cnn_out = nn.Sequential(nn.Linear(1024, 256), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(0.5))
        # RNN-based temporal network
        self.temporal_net = nn.GRU(256, 128, batch_first=True, dropout=0.3, bidirectional=True) 
        self.classifier = nn.Sequential(nn.Linear(256, 256), 
                                        nn.ReLU(inplace=True))              
        self.emo_regressor = nn.Sequential(nn.Linear(256, 256), 
                                           nn.ReLU(inplace=True))
        self.top_layer_class = nn.Linear(256, num_classes)
        self.top_layer_attri = nn.Linear(256, 1)
        self._initialize_weights()

    def forward(self, x):
        # shared CNN feature extraction model
        x = self.cnn_features(x)
        x = x.view(x.size(0), -1)
        x = self.cnn_out(x)
        # consider temporal info between chunks (using RNN-based model)        
        x = x.view(-1, 11, x.size(1))
        x, _ = self.temporal_net(x)
        x = x.reshape(-1, x.size(2))
        # for deep-cluster classification
        x_class = self.classifier(x)
        if self.top_layer_class:
            x_class = self.top_layer_class(x_class)
        # for emotion regression
        x_attri = self.emo_regressor(x)  
        x_attri = self.top_layer_attri(x_attri)
        return x_class, x_attri

    def _initialize_weights(self):
        for y,m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(input_dim, batch_norm):
    layers = []
    in_channels = input_dim
    cfg = [32, 32, '64', '64', '128', '128', '128']
    for v in cfg:
        # max pooling layer
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
        # conv2D with stride 2
        elif (type(v) == str)&(v!='M'):
            conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = int(v)                  
        
        # conv2D with stride 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(bn, out):
    inp_dim = 1
    model = VGG(make_layers(inp_dim, bn), out)
    return model
