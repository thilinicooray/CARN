'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm
import copy

from ..lib.attention import Attention
from ..lib.classifier import SimpleClassifier
from ..lib.fc import FCNet
import torchvision as tv
from ..utils import cross_entropy_loss

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn , torch.mean(scores,1)

class Top_Down_Baseline(nn.Module):
    def __init__(self, vgg_model, tda_model):
        super(Top_Down_Baseline, self).__init__()
        self.vgg_model = vgg_model
        self.tda_model = tda_model

    def forward(self, v_org, gt_verb):

        vgg_out = self.vgg_model(v_org)
        tda_out = self.tda_model(v_org, None)

        #print(verb_role_impact.size(), verb_role_impact[:5], qctx_out.size(), baseline_out.size())
        '''q_impact = verb_role_impact.unsqueeze(-1)
        b_impact = torch.ones(q_impact.size(0), q_impact.size(1), 1).cuda() - q_impact'''

        verb_pred = tda_out + vgg_out

        #role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return verb_pred







def build_vgg_tda_verb_joint(vgg_model, tda_model):



    return Top_Down_Baseline(vgg_model, tda_model)


