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
from itertools import chain


class CLIPDataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        "data parallel for clip model"
        super(CLIPDataParallel, self).__init__(module, device_ids, output_device, dim)

        #self.module=module

    def encode_image(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module.encode_image(*inputs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("Module must have its parameters and buffers"
                                   "on devive {} (device_ids[0]) but found one of"
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module.encode_image(*inputs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        replicas = [r.encode_image for r in replicas]
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


class CLIPimf(nn.Module):

    def __init__(self, model, preprocess):
        super(CLIPimf, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPDataParallel(model)
        self.convert_to_fp32()
        self.preprocess = preprocess
        #self.model.to(device)
        #self.model, self.preprocess = clip.load('ViT-B/32', self.device)
    
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    def forward(self, x, param_dict=None):
        x = self._flatten(x)
        #x = self.preprocess(x)
        #x = self.model.module.encode_image(x)
        x = self.model.encode_image(x)

        return x

    def convert_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()
            #p.grad.data = p.grad.data.float()

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



