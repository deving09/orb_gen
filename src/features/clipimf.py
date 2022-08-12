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




class TextEncorder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.tranformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16 # Number of Context Tokens cfg.TRAINER.N_CTX
        ctx_init = False # "a photo of a"   #cfg.TRAINER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        #clip_imsize = clip_model.visual.input_resolution

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0,  1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            pass
        else:
            # Random Initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')

        self.ctx = nn.Parameter(ctx_vectors)

        if True: #cfg.CoCoOp
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) #(n_cls, n_tkn)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        self.register_buffer("token_prefix", embedding[:, :1, :]) # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) # CLUS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts # torch.Tensor
        self.name_lens = name_lens


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix, # (dim0, 1, dim)
                ctx,    # (dim0, n_ctx, dim)
                suffix, # (dim0, *, dim)
                
            ],
            dim=1
        )
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        ctx = self.ctx                             # (n_ctx, ctx_dim)

        if True: #cfg.CoCoOp
            bias = self.meta_net(im_features)      # (batch, ctx_dim)
            bias = bias.unsqueeze(1)               # (batch, 1, ctx_dim)
            ctx = ctx_unsqueeze(0)                 # (1, n_ctx, ctx_dim)
            ctx_shifted = ctx + bias               # (batch, n_ctx, ctx_dim)

            prompts = []
            for ctx_shifted_i in ctx_shifted:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                pts_i = self.construct_prompts(ctx_i, prefix, suffix) # (n_cls, n_tkn, ctx_dim)
                prompts.append(pts_i)

            prompts = torch.stack(prompts)
            return prompts
        else:
            #prompts = []
            prompts = self.construct_prompts(ctx, prefix, suffix)
            return prompts


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # Replace with preprocess
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)

        #
        return logits



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



