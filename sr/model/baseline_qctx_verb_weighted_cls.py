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
    def __init__(self, baseline_model, qctx_model, classifier, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.baseline_model = baseline_model
        self.qctx_model = qctx_model
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v_org, gt_verb, verb_role_impact):

        baseline_out = self.baseline_model.forward_hiddenrep(v_org, gt_verb).contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)
        qctx_out = self.qctx_model.forward_hiddenrep(v_org, gt_verb).contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

        #print(verb_role_impact.size(), verb_role_impact[:5], qctx_out.size(), baseline_out.size())
        q_impact = verb_role_impact.unsqueeze(-1)
        b_impact = torch.ones(q_impact.size(0), q_impact.size(1), 1).cuda() - q_impact

        final_rep = q_impact * qctx_out + b_impact * baseline_out

        final_rep = self.classifier(final_rep.contiguous().view(v_org.size(0)* self.encoder.max_role_count, -1))

        role_label_pred = final_rep.contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())

        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3

        return loss


def build_baseline_qctx_joint(baseline_model, qctx_model, num_ans_classes, encoder):

    hidden_size = 1024

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Baseline(baseline_model, qctx_model, classifier, encoder)


