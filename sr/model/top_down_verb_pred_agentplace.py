'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

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
    def __init__(self, covnet, agent_emb, place_emb, agent_classifier, place_classifier, query_composer, v_att, q_net,
                 v_net, classifier, Dropout_C):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = covnet
        self.agent_emb = agent_emb
        self.place_emb = place_emb
        self.agent_classifier = agent_classifier
        self.place_classifier = place_classifier
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.Dropout_C = Dropout_C
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, v_org, agent_feat, place_feat):

        #get agent and place idx to form the query
        agent_idx = torch.max(self.agent_classifier(agent_feat),-1)[1]
        place_idx = torch.max(self.place_classifier(place_feat),-1)[1]
        agent_embd = self.agent_emb(agent_idx)
        place_embd = self.place_emb(place_idx)

        concat_query = torch.cat([ agent_embd, place_embd], -1)
        q_emb = self.Dropout_C(self.query_composer(concat_query))

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

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
        out = mfb_l2

        logits = self.classifier(out)

        return logits

    def calculate_verb_loss(self, verb_pred, gt_verbs):

        criterion = nn.CrossEntropyLoss()

        loss = criterion(verb_pred, gt_verbs.squeeze())

        return loss

def build_top_down_baseline_verb(agent_emb, place_emb, agent_classifier, place_classifier, num_ans_classes):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    agent_emb = agent_emb
    place_emb = place_emb
    agent_classifier = agent_classifier
    place_classifier = place_classifier
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    Dropout_C = nn.Dropout(0.3)

    return Top_Down_Baseline(covnet, agent_emb, place_emb, agent_classifier, place_classifier, query_composer, v_att, q_net,
                             v_net, classifier, Dropout_C)


