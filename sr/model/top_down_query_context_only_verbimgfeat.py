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

    return torch.matmul(p_attn, value), p_attn

class Top_Down_Baseline(nn.Module):
    def __init__(self, convnet, img_feat_combiner, role_emb, verb_emb, query_composer, v_att, q_net, v_net, neighbour_attention, updated_query_composer,
                 w_q, w_i, w_qc, w_prev, Dropout_C, classifier, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = convnet
        self.img_feat_combiner = img_feat_combiner
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.updated_query_composer = updated_query_composer
        self.neighbour_attention = neighbour_attention
        self.Dropout_C = Dropout_C
        self.w_q = w_q
        self.w_i = w_i
        self.w_qc = w_qc
        self.w_prev = w_prev
        self.classifier = classifier
        self.encoder = encoder

    def forward(self, v_org, img_feat, gt_verb):

        q_list = []
        ans_list = []
        n_heads = 1

        img_features_org = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features_org.size()

        img_features_org = img_features_org.view(batch_size, -1, conv_h* conv_w)
        img_features_org = img_features_org.permute(0, 2, 1)

        #img = F.relu(self.img_feat_combiner(torch.cat([img_features_org,img_feat], -1)))

        img = img_features_org.expand(self.encoder.max_role_count, img_features_org.size(0), img_features_org.size(1), img_features_org.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * self.encoder.max_role_count, -1, img.size(-1))

        role_idx = self.encoder.get_role_ids_batch(gt_verb)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        verb_embd = self.verb_emb(gt_verb)
        role_embd = self.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        # mask out non-existing roles from (max_role x max_role) adj. matrix
        mask = self.encoder.get_adj_matrix_noself(gt_verb)

        if torch.cuda.is_available():
            mask = mask.to(torch.device('cuda'))

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        #out = q_repr * v_repr
        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        q_list.append(q_repr)
        ans_list.append(out)

        #get answer with verb grid features
        img_v = img_feat.expand(self.encoder.max_role_count, img_feat.size(0), img_feat.size(1), img_feat.size(2))

        img_v = img_v.transpose(0,1)
        img_v = img_v.contiguous().view(batch_size * self.encoder.max_role_count, -1, img_v.size(-1))

        att_verb = self.v_att(img_v, q_emb)
        v_emb_verb = (att_verb * img_v).sum(1)
        v_repr_verb = self.v_net(v_emb_verb)

        #out_verb = q_repr * v_repr_verb

        mfb_iq_eltwise_verb = torch.mul(q_repr, v_repr_verb)

        mfb_iq_drop_verb = self.Dropout_C(mfb_iq_eltwise_verb)

        mfb_iq_resh_verb = mfb_iq_drop_verb.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
        mfb_iq_sumpool_verb = torch.sum(mfb_iq_resh_verb, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out_verb = torch.squeeze(mfb_iq_sumpool_verb)                     # N x 1000
        mfb_sign_sqrt_verb = torch.sqrt(F.relu(mfb_out_verb)) - torch.sqrt(F.relu(-mfb_out_verb))
        mfb_l2_verb = F.normalize(mfb_sign_sqrt_verb)
        out_verb = mfb_l2_verb


        for i in range(1):

            cur_group = out.contiguous().view(batch_size, self.encoder.max_role_count, -1)

            neighbours, _ = self.neighbour_attention(cur_group, cur_group, cur_group, mask=mask)

            withctx = neighbours.contiguous().view(batch_size* self.encoder.max_role_count, -1)

            updated_q_emb = self.Dropout_C(self.updated_query_composer(torch.cat([withctx,role_verb_embd], -1)))

            att = self.v_att(img, updated_q_emb)
            v_emb = (att * img).sum(1)
            v_repr = self.v_net(v_emb)
            q_repr = self.q_net(updated_q_emb)

            #out = q_repr * v_repr
            mfb_iq_eltwise = torch.mul(q_repr, v_repr)

            mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

            mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
            mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
            mfb_l2 = F.normalize(mfb_sign_sqrt)
            out = mfb_l2

            '''gate = torch.sigmoid(q_list[-1] * q_repr)
            out = gate * ans_list[-1] + (1-gate) * out + out_verb'''

            ctx_gate = torch.sigmoid(out + out_verb)
            out = self.Dropout_C((1-ctx_gate)* out_verb + ctx_gate * torch.tanh(out + ans_list[-1]))


            q_list.append(q_repr)
            ans_list.append(out)

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(batch_size, self.encoder.max_role_count, -1)

        return role_label_pred

    '''def calculate_loss(self, gt_verbs, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                for j in range(0, self.encoder.max_role_count):
                    frame_loss += cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.encoder.get_num_labels())
                frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                loss += frame_loss

        final_loss = loss/batch_size
        return final_loss'''

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
    n_heads = 2

    covnet = vgg16_modified()
    img_feat_combiner = weight_norm(nn.Linear(img_embedding_size * 2, img_embedding_size * 2), dim=None)
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    updated_query_composer = FCNet([hidden_size + word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    neighbour_attention = MultiHeadedAttention(4, hidden_size, dropout=0.1)
    Dropout_C = nn.Dropout(0.1)

    w_q = weight_norm(nn.Linear(hidden_size, hidden_size), dim=None)
    w_i = weight_norm(nn.Linear(hidden_size, hidden_size), dim=None)
    w_qc = weight_norm(nn.Linear(hidden_size, hidden_size), dim=None)
    w_prev = weight_norm(nn.Linear(hidden_size, hidden_size), dim=None)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes+1, 0.5)

    return Top_Down_Baseline(covnet, img_feat_combiner, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, neighbour_attention, updated_query_composer, Dropout_C, w_q, w_i, w_qc, w_prev, classifier, encoder)


