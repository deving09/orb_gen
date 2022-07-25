# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from features.resnet import resnet18, resnet18_84
from features.efficientnet import efficientnetb0
from features.clipimf import clip_vitb_32, clip_vitb_16, clip_resnet50, clip_resnet101

extractors = {
        'resnet18': resnet18,
        'resnet18_84': resnet18_84,
        'efficientnetb0' : efficientnetb0,
        'clip_vitb_32': clip_vitb_32,
        'clip_vitb_16': clip_vitb_16,
        'clip_resnet50': clip_resnet50,
        'clip_resnet101': clip_resnet101
        }
