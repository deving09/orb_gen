"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file resnet.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/resnet.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps/)

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import torch
import torch.nn as nn
from models.normalisation_layers import TaskNorm, get_normalisation_layer
from feature_adapters.resnet_adaptation_layers import FilmLayer, FilmLayerGenerator

import clip

class CLIPimf(nn.Module):

    def __init__(self):
        super(CLIPimf, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
    
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x):
        x = self._flatten(x)
        x = self.preprocess(image)

        x = self.model.encode_image(x)

        return x

    @property
    def output_size(self):
        return 512
        #pass


class ResNet(nn.Module):
    def __init__(self, block, layers, bn_fn, initial_pool=True, conv1_kernel_size=7):
        super(ResNet, self).__init__()
        self.initial_pool = initial_pool # False for 84x84
        self.inplanes = self.curr_planes = 64
        self.conv1 = nn.Conv2d(3, self.curr_planes, kernel_size=conv1_kernel_size, stride=2, padding=1, bias=False)
        self.bn1 = bn_fn(self.curr_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0], bn_fn)
        self.layer2 = self._make_layer(block, self.inplanes * 2, layers[1], bn_fn, stride=2)
        self.layer3 = self._make_layer(block, self.inplanes * 4, layers[2], bn_fn, stride=2)
        self.layer4 = self._make_layer(block, self.inplanes * 8, layers[3], bn_fn, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, TaskNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, bn_fn, stride=1):
        downsample = None
        if stride != 1 or self.curr_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.curr_planes, planes * block.expansion, stride),
                bn_fn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.curr_planes, planes, bn_fn, stride, downsample))
        self.curr_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.curr_planes, planes, bn_fn))

        return nn.Sequential(*layers)

    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x, param_dict=None):
        x = self._flatten(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    @property
    def output_size(self):
        return 512



def clip_vitb_32():
    pass

def resnet18(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    nl = get_normalisation_layer(batch_norm)
    if with_film:
        model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], nl, **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model

def resnet18_84(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **kwargs):
    """
        Constructs a ResNet-18 model for 84 x 84 images.
    """
    nl = get_normalisation_layer(batch_norm)
    if with_film:
        model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], nl, initial_pool=False, conv1_kernel_size=5, **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], nl, initial_pool=False, conv1_kernel_size=5, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model
