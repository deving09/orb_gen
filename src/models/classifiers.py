"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file adaptation_networks.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/adaptation_networks.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.mlps import DenseResidualBlock


import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()




class HeadClassifier(nn.Module):
    """
    Class for a head-style classifier which is created by a computation of (context) features. Similar to https://github.com/cambridge-mlg/cnaps.
    """
    def __init__(self):
        """
        Creates instance of HeadClassifier.
        :return: Nothing.
        """
        super().__init__()

    def _set_device(self, device):
        self.device = device
    
    def _build_class_reps(self, context_features, context_labels, ops_counter):
        class_reps = OrderedDict()
        for c in torch.unique(context_labels):
            t1 = time.time()
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = self._mean_pooling(class_features)
            class_reps[c.item()] = class_rep
            if ops_counter:
                torch.cuda.synchronize()
                ops_counter.log_time(time.time() - t1)
                ops_counter.add_macs(context_features.size(0)) # selecting class features
                ops_counter.add_macs(class_features.size(0) * class_features.size(1)) # mean pooling

        return class_reps
    
    def reset(self):
        self.param_dict = {}

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
    
    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)
 
class VersaClassifier(HeadClassifier):
    """
    Class for a Versa classifier (https://github.com/cambridge-mlg/cnaps). Context features are passed through two hyper-networks to generate the weight and bias parameters of a linear classification layer, respectively.
    """
    def __init__(self, in_size):
        """
        Creates instance of VersaClassifier.
        :param in_size: (int) Size of feature extractor output.
        :return: Nothing.
        """
        super().__init__()
        self.weight_processor = self._make_layer(in_size, in_size)
        self.bias_processor = self._make_layer(in_size, 1)
        self.param_dict = {}

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseResidualBlock(in_size, out_size)
    
    def predict(self, target_features):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param target_features: (torch.Tensor) Batch of target features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """ 
        return F.linear(target_features, self.param_dict['weight'], self.param_dict['bias'])

    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that passes per-class context features through two hyper-networks to generate the weight vector and bias scalar for each class in a linear classification layer.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 

        class_rep_dict = self._build_class_reps(context_features, context_labels, ops_counter)
        class_weight = []
        class_bias = []

        label_set = list(class_rep_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            t1 = time.time()
            nu = class_rep_dict[class_num]
            class_weight.append(self.weight_processor(nu))
            class_bias.append(self.bias_processor(nu))
            if ops_counter:
                torch.cuda.synchronize()
                ops_counter.log_time(time.time() - t1)
                ops_counter.compute_macs(self.weight_processor, nu)
                ops_counter.compute_macs(self.bias_processor, nu)


        self.param_dict['weight'] = torch.cat(class_weight, dim=0)
        self.param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])
        
class PrototypicalClassifier(HeadClassifier):
    """
    Class for a ProtoNets classifier (https://github.com/jakesnell/prototypical-networks). Context features are averaged per class to obtain the weight parameters of a linear classification layer.
    """
    def __init__(self):
        """
        Creates instance of PrototypicalClassifier.
        :return: Nothing.
        """
        super().__init__()
        self.param_dict = {}

    def predict(self, target_features):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param target_features: (torch.Tensor) Batch of target features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """ 
        return F.linear(target_features, self.param_dict['weight'], self.param_dict['bias'])

    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that computes the per-class mean of context features and sets this as the class weight vector in a linear classification layer.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 

        class_rep_dict = self._build_class_reps(context_features, context_labels, ops_counter)
        class_weight = []
        class_bias = []

        label_set = list(class_rep_dict.keys())
        label_set.sort()
        num_classes = len(label_set)

        for class_num in label_set:
            t1 = time.time()
            # equation 8 from the prototypical networks paper
            nu = class_rep_dict[class_num]
            class_weight.append(2 * nu)
            class_bias.append((-torch.matmul(nu, nu.t()))[None, None])
            if ops_counter:
                torch.cuda.synchronize()
                ops_counter.log_time(time.time() - t1)
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # 2* in class weight
                ops_counter.add_macs(nu.size(0)**2 * nu.size(1)) # matmul in  class bias
                ops_counter.add_macs(nu.size(0) * nu.size(1)) # -1* in  class bias

        self.param_dict['weight'] = torch.cat(class_weight, dim=0)
        self.param_dict['bias'] = torch.reshape(torch.cat(class_bias, dim=1), [num_classes, ])
        


class CLIPLinearClassifier(nn.Module):
    """
    Class for a linear based on CLIP model
    """
    
    def __init__(self, in_size, clip_model):
        """
        Creates instance of CLIPLinearClassifier.
        :param in_size (int) Size of Feature extractor output.
        :return: Nothing.
        """
        super().__init__()
        self.in_size = in_size
        self._clip_model = clip_model
        pass

    def _set_device(self, device):
        self.device = device


    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that creates and initialises a linear classification layer based on text
        prompts
        """

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in object_list]).to(self.device)
        #self._clip_model.to(self.device)
        text_features = self._clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        n_cls = len(object_list)
        self.linear = nn.Linear(self.in_size, n_cls, bias=True) 
        
        nn.init.kaiming_uniform_(self.linear.weight, mode="fan_out")
        nn.init.zeros_(self.linear.bias)
        self.linear.weight.data = text_features #context_features.dtype)
        #self.linear.bias.data = self.linear.bias.type(context_features.dtype)
        self.linear.to(self.device)
        #self._clip_model.to(self.device)


    def predict(self, features, ops_counter=None):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param features: (torch.Tensor) Batch of features.
        :return: (torch.Tensor) Logits over object classes for each feature.
        """
        t1 = time.time()

        features /= features.norm(dim=-1, keepdim=True)

        out = self.linear(features)
        if ops_counter:
            torch.cuda.synchronize()
            ops_counter.log_time(time.time() - t1)
            ops_counter.compute_macs(self.linear, features)
        
        return out

    def reset(self):
        self.linear = None


class LinearClassifier(HeadClassifier):
    """
    Class for a linear classification layer.
    """
    def __init__(self, in_size): #, device=None):
        """
        Creates instance of LinearClassifier.
        :param in_size: (int) Size of feature extractor output.
        :return: Nothing.
        """ 
        super().__init__()
        self.in_size = in_size
        #self.device = device

    def _set_device(self, device):
        self.device = device

    #def configure(self, out_size, device, init_zeros=True):
    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that creates and initialises a linear classification layer.
        :param out_size: (int) Number of classes in classification layer.
        :param device: (torch.device) Device to move classification layer to.
        :init_zeros: (bool) If True, initialise classification layer with zeros, otherwise use Kaiming uniform.
        :return: Nothing.
        """
        n_cls = len(torch.unique(context_labels))
        self.linear = nn.Linear(self.in_size, n_cls, bias=True) 
       
        
        nn.init.kaiming_uniform_(self.linear.weight, mode="fan_out")
        nn.init.zeros_(self.linear.bias)
        self.linear.weight.data = self.linear.weight.type(context_features.dtype)
        self.linear.bias.data = self.linear.bias.type(context_features.dtype)
        self.linear.to(self.device)
  
    def predict(self, features, ops_counter=None):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param features: (torch.Tensor) Batch of features.
        :return: (torch.Tensor) Logits over object classes for each feature.
        """
        t1 = time.time()

        out = self.linear(features)
        #if ops_counter:
        #    torch.cuda.synchronize()
        #    ops_counter.log_time(time.time() - t1)
        #    ops_counter.compute_macs(self.linear, features)
        
        return out

    def reset(self):
        self.linear = None



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        #self.transformer = clip_model.module.transformer
        self.positional_embedding = clip_model.positional_embedding
        #self.positional_embedding = clip_model.module.positional_embedding
        self.ln_final = clip_model.ln_final
        #self.ln_final = clip_model.module.ln_final
        self.text_projection = clip_model.text_projection
        #self.text_projection = clip_model.module.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def _set_device(self, device):
        print("Set Text Device: %s" % device)
        self.device = device
        self.transformer.to(device)
        self.text_projection = self.text_projection.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        self.ln_final.to(device)
        #pass


class PromptLearner(nn.Module):
    
    def __init__(self, prompt_meth, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16 # Number of Context Tokens cfg.TRAINER.N_CTX
        ctx_init = False # "a photo of a"   #cfg.TRAINER.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        self.prompt_meth = prompt_meth
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
            #ctx_vectors = ctx_vectors.to(device)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Number of context words (tokens): {n_ctx}')

        self.ctx = nn.Parameter(ctx_vectors)

        if prompt_meth == "cocoop": #cfg.CoCoOp
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
            ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.clip.tokenize(p) for p in prompts]) #(n_cls, n_tkn)
        tokenized_prompts = tokenized_prompts.to(device)

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

        if self.prompt_meth == "cocoop":           #True: #cfg.CoCoOp
            bias = self.meta_net(im_features)      # (batch, ctx_dim)
            bias = bias.unsqueeze(1)               # (batch, 1, ctx_dim)
            ctx = ctx.unsqueeze(0)                 # (1, n_ctx, ctx_dim)
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
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
            prompts = self.construct_prompts(ctx, prefix, suffix)
            return prompts





class CLIPPromptClassifier(HeadClassifier):
    """
    Class for a Context Optimization based prompt classifier ().
    Initializes Randomly or with base prompts and can be either a global prompt or a contextual prompt
    """

    def __init__(self, in_size, clip_model, meth="coop"):
        """
        """
        super().__init__()
        self.in_size = in_size
        self._clip_model = clip_model.module
        #self._clip_model = clip_model
        self.meth = meth
        self.text_encoder = TextEncoder(self._clip_model)
        self.logit_scale = self._clip_model.logit_scale
        self.dtype = self._clip_model.dtype
        self.prompt_learner = None


    def _set_device(self, device):
        #self._clip_model.to(device)
        self.device = device
        self.text_encoder._set_device(self.device)
        if self.prompt_learner:
            self.prompt_learner.to(self.device)


    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that creates and initialises a linear classification layer based on learned
        prompts
        """
        #self._clip_model.to(self.device)
        self.prompt_learner = PromptLearner(self.meth, object_list, self._clip_model, self.device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.prompt_learner.to(self.device)


    def predict(self, features, ops_counter=None):
        """
        Function that passes a batch of target features through linear classification layer to get logits over object classes for each feature.
        :param features: (torch.Tensor) Batch of features.
        :return: (torch.Tensor) Logits over object classes for each feature.
        """
        t1 = time.time()

        #self._set_device(features.get_device())

        #features = features.to(self.device)

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner(features)
        
        if self.meth == "cocoop":
            logits = []
            for pts_i, imf_i in zip(prompts, features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)

            logits = torch.stack(logits)
            
            return logits
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            logits = logit_scale * features @ text_features.t()
            #print(features.get_device())
            #print(text_features.get_device())
            #logits = features @ text_features.t()
            return logits


        

    def reset(self):
        self.prompt_learner = None





class MahalanobisClassifier(HeadClassifier):
    """
    Class for a Mahalanobis classifier (https://github.com/peymanbateni/simple-cnaps). Computes per-class distributions using context features. Target features are classified by the shortest Mahalanobis distance to these distributions.
    """
    def __init__(self):
        """
        Creates instance of MahalanobisClassifier.
        :return: Nothing.
        """
        super().__init__()
        self.param_dict = {}

    def configure(self, context_features, context_labels, ops_counter=None, object_list=None):
        """
        Function that computes a per-class distribution (mean, precision) using the context features.
        :param context_features: (torch.Tensor) Context features.
        :param context_labels: (torch.Tensor) Corresponding class labels for context features.
        :param ops_counter: (utils.OpsCounter or None) Object that counts operations performed.
        :return: Nothing.
        """
        assert context_features.size(0) == context_labels.size(0), "context features and labels are different sizes!" 

        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(context_features, ops_counter)
        
        for c in torch.unique(context_labels):
            t1 = time.time()
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            means.append(self._mean_pooling(class_features).squeeze())
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            class_covariance_estimate = self._estimate_cov(class_features, ops_counter)
            covariance_matrix = (lambda_k_tau * class_covariance_estimate) \
                                + ((1 - lambda_k_tau) * task_covariance_estimate) \
                                + torch.eye(class_features.size(1), device=class_features.device)
            precisions.append(torch.inverse(covariance_matrix))

            if ops_counter:
                torch.cuda.synchronize()
                ops_counter.log_time(time.time() - t1)
                ops_counter.add_macs(context_features.size(0)) # selecting class features
                ops_counter.add_macs(class_features.size(0) * class_features.size(1)) # mean pooling
                ops_counter.add_macs(1) # computing lambda_k_tau
                ops_counter.add_macs(class_covariance_estimate.size(0) * class_covariance_estimate.size(1)) # lambda_k_tau * class_covariance_estimate
                ops_counter.add_macs(task_covariance_estimate.size(0) * task_covariance_estimate.size(1)) # (1-lambda_k_tau) * task_covariance_estimate
                ops_counter.add_macs(1/3*covariance_matrix.size(0) ** 3 + covariance_matrix.size(0) ** 2 - 4/3*covariance_matrix.size(0)) # computing inverse of covariance_matrix, taken from https://en.wikipedia.org/wiki/Gaussian_elimination#Computational_efficiency
                # note, sum of 3 matrices to compute covariance_matrix is not included here

        self.param_dict['means'] = (torch.stack(means))
        self.param_dict['precisions'] = (torch.stack(precisions))
         
    def predict(self, target_features):
        """
        Function that processes a batch of target features to get logits over object classes for each feature. Target features are classified by their Mahalanobis distance to the class means including the class precisions.
        :param target_features: (torch.Tensor) Batch of target features.
        :return: (torch.Tensor) Logits over object classes for each target feature.
        """ 
        # grabbing the number of classes and query examples for easier use later in the function
        number_of_classes = self.param_dict['means'].size(0)
        number_of_targets = target_features.size(0)

        # calculate the Mahalanobis distance between target features and the class means including the class precision
        repeated_target = target_features.repeat(1, number_of_classes).view(-1, self.param_dict['means'].size(1))
        repeated_class_means = self.param_dict['means'].repeat(number_of_targets, 1)
        repeated_difference = (repeated_class_means - repeated_target)
        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes,
                                                       repeated_difference.size(1)).permute(1, 0, 2)
        first_half = torch.matmul(repeated_difference, self.param_dict['precisions'])
        logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

        return logits

    @staticmethod
    def _estimate_cov(examples, ops_counter, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
        Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
        Returns:
            The covariance matrix of the variables.
        """
        t1 = time.time()
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        cov_matrix = factor * examples.matmul(examples_t)

        if ops_counter:
            torch.cuda.synchronize()
            ops_counter.log_time(time.time() - t1)
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing mean
            ops_counter.add_macs(1) # computing factor
            ops_counter.add_macs(examples.size(0)**2 * examples.size(1)) # computing matmul
            ops_counter.add_macs(examples.size(0) * examples.size(1)) # computing factor*cov_matrix

        return cov_matrix
