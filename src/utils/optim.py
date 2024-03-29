# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
optimizers = {
        'adam': torch.optim.Adam,
        'sgd' : torch.optim.SGD
        }

def cross_entropy(test_logits, test_labels, reduction='mean'):
    return F.cross_entropy(test_logits, test_labels, reduction=reduction)
    
def init_optimizer(model, lr, optimizer_type='adam', extractor_scale_factor=1.0, frozen=False):
    if frozen:
        feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params, model.parameters())
        optimizer_fn = optimizers[optimizer_type]
        optimizer = optimizer_fn([
                            {'params': base_params }
                            ], lr=lr, momentum=0.9)#, weight_decay=0.001)
                            #], lr=lr)
    else:
        feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
        base_params = list(filter(lambda p: id(p) not in feature_extractor_params, model.parameters()))
        optimizer_fn = optimizers[optimizer_type]
        optimizer = optimizer_fn([
                            {'params': base_params },
                            {'params': model.feature_extractor.parameters(), 'lr': lr*extractor_scale_factor}
                            ], lr=lr, momentum=0.9)#, weight_decay=0.001)
                            #], lr=lr)
    optimizer.zero_grad()
    return optimizer
