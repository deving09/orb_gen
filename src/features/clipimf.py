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

from memory_profiler import profile


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
        #print(self.device_ids)
        #print([i[0].device for i in inputs])
        if len(self.device_ids) == 1:
            return self.module.encode_image(*inputs[0])

        replicas = self.replicate(self.module.visual, self.device_ids[:len(inputs)])
        #print("Len Replicas: %d" % len(replicas))
        #print("Len Inputs: %d" % len(inputs))
        """
        for i, r in enumerate(replicas):
            if i == 0:
                print([p.device for p in r.parameters()])
            else:
                print(list(r.__dict__.keys()))
                print(list(r._former_parameters))
                print(["%s:%s" %(n, v.device) for n, v in r._former_parameters.items()])
                print(["%s:%s" %(n, v.device) for n, v in r._buffers.items()])
                print(["%s:%s" %(n, v.device) for n, v in r._parameters.items()])
                print(r.training)
                print(r._is_replica)
                print(r.class_embedding.device)
                print(r.positional_embedding.device)
                print(r.proj.device)
                #print(r._modules)
                print([b for b in r._modules])
                print(r._modules.__dict__)
                print(r._modules["transformer"]._modules["resblocks"]._modules)
                #print(r.proj.device)
                ['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules', 'input_resolution', 'output_dim', '_is_replica', '_former_parameters', 'class_embedding', 'positional_embedding', 'proj'
        """

        #replicas = [r.encode_image for r in replicas]
        #print(replicas)
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
    
    
    def convert_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()
            #p.grad.data = p.grad.data.float()

    @property
    def output_size(self):
        return 512
    
    
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x



class ImageCLIP(nn.Module):

    #@profile(precision=4)
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    #@profile(precision=4)
    def forward(self, image):
        #print(image.device)
        #image = self._flatten(image)
        return self.model.encode_image(image)

    #@profile(precision=4)
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x


class TextCLIP(nn.Module):

    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model


    def forward(self, text):

        return self.model(text)



class CLIPimf(nn.Module):
    
    #@profile(precision=4)
    def __init__(self, model, preprocess):
        super(CLIPimf, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model # CLIPDataParallel(model)
        #self.model = CLIPDataParallel(model)
        self.convert_to_fp32()
        self.model = nn.DataParallel(ImageCLIP(self.model))  #CLIPDataParallel
        
        #self.text_model = nn.DataParallel(TextCLIP(self.model))
        #self.model = nn.DataParallel(self.model)  #CLIPDataParallel
        self.preprocess = preprocess
        #self.model.to(device)
        #self.model, self.preprocess = clip.load('ViT-B/32', self.device)
    
    #@profile(precision=4)
    def _flatten(self, x):
        sz = x.size()
        return x.view(-1, sz[-3], sz[-2], sz[-1]) if x.dim() >=5 else x

    #@profile(precision=4)
    def forward(self, x, param_dict=None):
        x = self._flatten(x)
        #x = self.preprocess(x)
        #x = self.model.module.encode_image(x)
        #x = self.model.encode_image(x)
        x = self.model(x)

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
    model, preprocess = clip.load('ViT-B/32', jit=False) #, device)
    clip_model = CLIPimf(model, preprocess)
    #return  CLIPDataParallel(clip_model)
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



