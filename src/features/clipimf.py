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

    def __init__(self, model, preprocess):
        super(CLIPimf, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model
        self.preprocess = preprocess
        #self.model, self.preprocess = clip.load('ViT-B/32', self.device)
    
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x, param_dict=None):
        x = self._flatten(x)
        #x = self.preprocess(x)
        x = self.model.encode_image(x)

        return x

    @property
    def output_size(self):
        return 512
        #pass


def clip_vitb_32(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    clip_model = CLIPimf(model, preprocess)
    return clip_model


def clip_vitb_16(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/16', device)
    clip_model = CLIPimf(model, preprocess)
    return clip_model


def clip_resnet50(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    clip_model = CLIPimf(model, preprocess)
    return clip_model


def clip_resnet101(**kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN101', device)
    clip_model = CLIPimf(model, preprocess)
    return clip_model




"""
def resnet18(pretrained=False, pretrained_model_path=None, batch_norm='basic', with_film=False, **kwargs):
    nl = get_normalisation_layer(batch_norm)
    if with_film:
        model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], nl, **kwargs)
    else:
        model = ResNet(BasicBlock, [2, 2, 2, 2], nl, **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])

    return model
"""
