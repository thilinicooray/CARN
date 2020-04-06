'''
PyTorch implementation of GGNN based SR : https://arxiv.org/abs/1708.04320
GGNN implementation adapted from https://github.com/chingyaoc/ggnn.pytorch
'''

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import math
import copy


from ..lib.attention import Attention
from ..lib.fc import FCNet

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.out_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        self.vgg_classifier = nn.Sequential(*features) # Replace the model classifier

        self.resize = nn.Sequential(
            nn.Linear(4096, 1024)
        )

    def forward(self,x):
        features = self.vgg_features(x)
        y = self.resize(self.vgg_classifier(features.view(-1, 512*7*7)))
        return y

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
        x, self.attn, mean_scores = attention(query, key, value, mask=mask,
                                              dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), torch.mean(self.attn, 1)


class GNN_new(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_node,  n_steps):
        super(GNN_new, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps


        #neighbour projection
        self.W_p = nn.Linear(state_dim, state_dim)

        self.JOINT_EMB_SIZE = 5 * state_dim
        self.Linear_nodeproj = nn.Linear(state_dim, self.JOINT_EMB_SIZE)
        self.Linear_neighbourproj = nn.Linear(state_dim, self.JOINT_EMB_SIZE)

        #global projection
        '''self.W_g = nn.Linear(state_dim, state_dim)
        self.v_att = Attention(state_dim, state_dim, state_dim)
        self.q_net = FCNet([state_dim, state_dim ])
        self.v_net = FCNet([state_dim, state_dim ])'''


    def forward(self, current_nodes, mask):

        hidden_state = current_nodes

        for t in range(self.n_steps):

            # calculating neighbour info
            neighbours = hidden_state.contiguous().view(mask.size(0), self.n_node, -1)
            neighbours = neighbours.expand(self.n_node, neighbours.size(0), neighbours.size(1), neighbours.size(2))
            neighbours = neighbours.transpose(0,1)

            neighbours = neighbours * mask.unsqueeze(-1)
            neighbours = self.W_p(neighbours)
            neighbours = torch.sum(neighbours, 2)
            neighbours = neighbours.contiguous().view(mask.size(0)*self.n_node, -1)

            data_out = self.Linear_nodeproj(hidden_state)                   # data_out (batch, 5000)
            img_feature = self.Linear_neighbourproj(neighbours)      # img_feature (batch, 5000)
            iq = torch.mul(data_out, img_feature)
            iq = F.dropout(iq, 0.1, training=self.training)
            iq = iq.view(-1, 1, self.state_dim, 5)
            iq = torch.squeeze(torch.sum(iq, 3))                        # sum pool
            iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))       # signed sqrt
            hidden_state = F.normalize(iq)

        return hidden_state

class GGNN_Baseline(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, ggnn, classifier, encoder):
        super(GGNN_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.ggnn = ggnn
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v_org, gt_verb):

        img_features = self.convnet(v_org)

        v = img_features

        batch_size = v.size(0)

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        # repeat single image for max role count a frame can have
        img = v

        img = img.expand(self.encoder.max_role_count, img.size(0), img.size(1))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1)

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)
        role_embd = role_embd.view(batch_size * self.encoder.max_role_count, -1)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        verb_embed_expand = verb_embed_expand.contiguous().view(batch_size * self.encoder.max_role_count, -1)

        input2ggnn = img * role_embd * verb_embed_expand

        #mask out non exisiting roles from max role count a frame can have
        mask = self.encoder.get_adj_matrix_noself(gt_verb)
        if torch.cuda.is_available():
            mask = mask.to(torch.device('cuda'))

        out = self.ggnn(input2ggnn, mask)

        logits = self.classifier(out)

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

def build_ggnn_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, hidden_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, hidden_size)
    ggnn = GNN_new(state_dim = hidden_size, n_node=encoder.max_role_count,
                n_steps=4)
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(hidden_size, num_ans_classes)
    )

    return GGNN_Baseline(covnet, role_emb, verb_emb, ggnn, classifier, encoder)