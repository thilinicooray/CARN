'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import numpy as np

from ..lib.attention import Attention
from ..lib.classifier import SimpleClassifier
from ..lib.fc import FCNet
import torchvision as tv


class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features

class Top_Down_Baseline(nn.Module):
    def __init__(self, covnet, role_module, label_emb, query_composer, v_att, q_net,
                 v_net, resize_img_flat, classifier, Dropout_C):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = covnet
        self.role_module = role_module
        self.label_emb = label_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.resize_img_flat = resize_img_flat
        self.classifier = classifier
        self.Dropout_C = Dropout_C
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, v_org, verb_pred):

        #get agent and place idx to form the query
        #verb_pred = torch.max(self.cnn_verb_module(v_org),-1)[1].squeeze()

        agent_place_pred, agent_place_rep = self.role_module.forward_agentplace_noverb(v_org, verb_pred)

        role_rep_combo = torch.sum(agent_place_rep, 1)

        label_idx = torch.max(agent_place_pred,-1)[1].squeeze()
        agent_embd = self.label_emb(label_idx[:,0])
        place_embd = self.label_emb(label_idx[:,1])
        concat_query = torch.cat([ agent_embd, place_embd], -1)
        q_emb = self.Dropout_C(self.query_composer(concat_query))

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()
        img_feat_flat = self.avg_pool(img_features)
        img_feat_flat = self.resize_img_flat(img_feat_flat.squeeze())
        ext_ctx = img_feat_flat * role_rep_combo

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size, 1, -1, 1)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2 + ext_ctx

        logits = self.classifier(out)

        return logits

    def calculate_verb_loss(self, verb_pred, gt_verbs):

        criterion = nn.CrossEntropyLoss()

        loss = criterion(verb_pred, gt_verbs.squeeze())

        return loss

def build_top_down_baseline_verb(num_labels, num_ans_classes, role_module):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_module = role_module
    label_emb = nn.Embedding(num_labels + 1, word_embedding_size, padding_idx=num_labels)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    resize_img_flat = nn.Linear(img_embedding_size, hidden_size)
    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    Dropout_C = nn.Dropout(0.1)

    return Top_Down_Baseline(covnet, role_module, label_emb, query_composer, v_att, q_net,
                             v_net, resize_img_flat, classifier, Dropout_C)


