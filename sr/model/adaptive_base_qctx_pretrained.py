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
    def __init__(self, baseline_model, qctx_model,  v_net, avg_pool, resize_img_flat, reconstruct_img, proj1, proj2, classifier, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.baseline_model = baseline_model
        self.qctx_model = qctx_model
        self.v_net = v_net
        self.avg_pool = avg_pool
        self.resize_img_flat = resize_img_flat
        self.reconstruct_img = reconstruct_img
        self.proj1 = proj1
        self.proj2 = proj2
        self.classifier = classifier
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)

    def forward1(self, v_org, gt_verb):

        baseline_vatt = self.baseline_model.forward_hiddenrep(v_org, gt_verb)
        qctx_vatt = self.qctx_model.forward_hiddenrep(v_org, gt_verb)

        baseline_rep = self.v_net(baseline_vatt)
        qctx_rep = self.v_net(qctx_vatt)



        baseline_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(baseline_rep.size(0)).cuda(),
                                                                 self.proj1(baseline_rep).squeeze()).unsqueeze(-1)))

        qctx_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(qctx_rep.size(0)).cuda(),
                                                                 self.proj1(qctx_rep).squeeze()).unsqueeze(-1)))



        baseline_confidence_norm = baseline_confidence / (baseline_confidence + qctx_confidence)
        qctx_confidence_norm = qctx_confidence / (baseline_confidence + qctx_confidence)


        out = baseline_confidence_norm * baseline_rep + qctx_confidence_norm * qctx_rep

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def forward(self, v_org, gt_verb):

        baseline_vatt = self.baseline_model.forward_hiddenrep(v_org, gt_verb)
        qctx_vatt = self.qctx_model.forward_hiddenrep(v_org, gt_verb)

        img_features = self.qctx_model.convnet(v_org) + self.baseline_model.convnet(v_org)
        img_feat_flat = self.avg_pool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())

        baseline_rep = self.v_net(baseline_vatt)
        qctx_rep = self.v_net(qctx_vatt)

        baseline_out_frame = baseline_rep.contiguous().view(v_org.size(0), -1)
        qctx_out_frame = qctx_rep.contiguous().view(v_org.size(0), -1)


        recon_img_baseline = self.reconstruct_img(baseline_out_frame)
        recon_img_qctx = self.reconstruct_img(qctx_out_frame)

        print(recon_img_baseline.size(), img_feat_flat.size())


        baseline_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(baseline_rep.size(0)).cuda(),
                                                                 self.proj1(recon_img_baseline * img_feat_flat).squeeze()).unsqueeze(-1)))

        qctx_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(qctx_rep.size(0)).cuda(),
                                                             self.proj1(recon_img_qctx * img_feat_flat).squeeze()).unsqueeze(-1)))

        baseline_confidence_norm = baseline_confidence / (baseline_confidence + qctx_confidence)
        qctx_confidence_norm = qctx_confidence / (baseline_confidence + qctx_confidence)

        baseline_confidence_norm = baseline_confidence_norm.expand(self.encoder.max_role_count, baseline_confidence_norm.size(0), baseline_confidence_norm.size(1))
        baseline_confidence_norm = baseline_confidence_norm.transpose(0,1)
        baseline_confidence_norm = baseline_confidence_norm.contiguous().view(baseline_confidence_norm.size(0) * self.encoder.max_role_count, -1)

        qctx_confidence_norm = qctx_confidence_norm.expand(self.encoder.max_role_count, qctx_confidence_norm.size(0), qctx_confidence_norm.size(1))
        qctx_confidence_norm = qctx_confidence_norm.transpose(0,1)
        qctx_confidence_norm = qctx_confidence_norm.contiguous().view(qctx_confidence_norm.size(0) * self.encoder.max_role_count, -1)


        out = baseline_confidence_norm * baseline_rep + qctx_confidence_norm * qctx_rep

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

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



def build_adaptive_base_qctx(num_ans_classes, encoder, baseline_model, qctx_model):

    hidden_size = 1024
    img_embedding_size = 512

    v_net = FCNet([hidden_size, hidden_size])

    avg_pool = nn.AdaptiveAvgPool2d(1)
    resize_img_flat = nn.Linear(img_embedding_size, 1024)
    reconstruct_img = FCNet([hidden_size*6, hidden_size])

    proj1 = nn.Linear(hidden_size,1)
    proj2 = nn.Linear(1,1)


    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Baseline(baseline_model, qctx_model,  v_net, avg_pool, resize_img_flat, reconstruct_img, proj1, proj2, classifier, encoder)


