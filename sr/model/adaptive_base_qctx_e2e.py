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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)

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

    return torch.matmul(p_attn, value), p_attn

class Top_Down_Baseline(nn.Module):
    def __init__(self, covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                 v_net, neighbour_attention, updated_query_composer, Dropout_C,
                 v_net_joint, proj1, proj2, classifier, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = covnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.updated_query_composer = updated_query_composer
        self.neighbour_attention = neighbour_attention
        self.Dropout_C = Dropout_C
        self.classifier = classifier
        self.encoder = encoder
        self.v_net_joint = v_net_joint
        self.proj1 = proj1
        self.proj2 = proj2

    def forward(self, v_org, gt_verb):

        role_oh_encoding = self.encoder.get_verb2role_encoing_batch(gt_verb)
        if torch.cuda.is_available():
            role_oh_encoding = role_oh_encoding.to(torch.device('cuda'))

        mask = self.encoder.get_adj_matrix_noself(gt_verb)

        if torch.cuda.is_available():
            mask = mask.to(torch.device('cuda'))

        q_list = []
        ans_list = []
        n_heads = 1

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, v.size(2))

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, 1)   # N x 1 x 1000 x n_heads # we go with 1
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        base_out = mfb_l2
        #calc confidence
        #baseline_rep = self.v_net_joint(base_out)
        #baseline_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(baseline_rep.size(0)).cuda(),
                                                                 #self.proj1(baseline_rep).squeeze()).unsqueeze(-1)))


        #QCTX


        cur_group = base_out.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        neighbours, _ = self.neighbour_attention(cur_group, cur_group, cur_group, mask=mask)

        withctx = neighbours.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)

        updated_q_emb = self.Dropout_C(self.updated_query_composer(torch.cat([withctx, role_verb_embd], -1)))

        att = self.v_att(img, updated_q_emb)
        v_emb = (att * img).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(updated_q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        qctx_out = mfb_l2

        #calc confidence
        '''qctx_rep = self.v_net_joint(qctx_out)
        qctx_confidence = torch.sigmoid(self.proj2(torch.max(torch.zeros(qctx_rep.size(0)).cuda(),
                                                             self.proj1(qctx_rep).squeeze()).unsqueeze(-1)))



        baseline_confidence_norm = baseline_confidence / (baseline_confidence + qctx_confidence)
        qctx_confidence_norm = qctx_confidence / (baseline_confidence + qctx_confidence)


        out = baseline_confidence_norm * baseline_rep + qctx_confidence_norm * qctx_rep'''

        logits = self.classifier(qctx_out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

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

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), torch.mean(self.attn, 1)

def build_top_down_query_context_only_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    updated_query_composer = FCNet([hidden_size + word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    neighbour_attention = MultiHeadedAttention(4, hidden_size, dropout=0.1)
    Dropout_C = nn.Dropout(0.1)

    #confidence
    v_net_joint = FCNet([hidden_size, hidden_size])
    proj1 = nn.Linear(hidden_size,1)
    proj2 = nn.Linear(1,1)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Baseline(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, neighbour_attention, updated_query_composer, Dropout_C,
                             v_net_joint, proj1, proj2, classifier, encoder)


