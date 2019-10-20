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

class Contextualized_Reasoner_Full(nn.Module):
    def __init__(self, convnet, img_refiner, role_emb, verb_emb, query_composer, updated_query_composer,
                 v_att, q_net, v_net, neighbour_attention, resize_ctx, Dropout_C, flattened_ctx_img, classifier, encoder):
        super(Contextualized_Reasoner_Full, self).__init__()
        self.convnet = convnet
        self.img_refiner = img_refiner
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.updated_query_composer = updated_query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.resize_ctx = resize_ctx
        self.neighbour_attention = neighbour_attention
        self.Dropout_C = Dropout_C
        self.classifier = classifier
        self.flattened_ctx_img = flattened_ctx_img
        self.encoder = encoder

    def forward(self, v_org, gt_verb):

        q_list = []
        v_list = []
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

        img = self.img_refiner(img)

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
        v_list.append(v_repr)
        ans_list.append(out)

        for i in range(1):

            cur_group = out.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            neighbours, _ = self.neighbour_attention(cur_group, cur_group, cur_group, mask=mask)

            withctx = neighbours.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)

            # context aware img reasoning

            withctx_expand = withctx.expand(img.size(1), withctx.size(0), withctx.size(1))
            withctx_expand = withctx_expand.transpose(0,1)
            # combine neighbour information with all regions of the image
            added_img = torch.cat([withctx_expand, img], -1)
            added_img = added_img.contiguous().view(-1, added_img.size(-1))
            # use a gating mechanism to decide how much information is necessary from each region
            # based on context information to answer current query
            added_img = torch.sigmoid(self.Dropout_C(self.resize_ctx(added_img)))
            added_img = added_img.contiguous().view(v.size(0) * self.encoder.max_role_count, -1, added_img.size(-1))
            # update regions using the gate
            # can we mask out areas of context, so what is remaining is what we want
            updated_img = added_img * img

            # context aware query reasoning
            updated_q_emb = self.Dropout_C(self.updated_query_composer(torch.cat([withctx,role_verb_embd], -1)))

            att_q = self.v_att(updated_img, updated_q_emb)
            v_emb_q = (att_q * updated_img).sum(1)
            v_repr_q = self.v_net(v_emb_q)
            q_repr_q = self.q_net(updated_q_emb)

            ctx_aware_query_out = torch.mul(q_repr_q, v_repr_q)

            combined_out = ctx_aware_query_out

            mfb_iq_drop_t = self.Dropout_C(combined_out)

            mfb_iq_resh_t = mfb_iq_drop_t.view(batch_size* self.encoder.max_role_count, 1, -1, n_heads)   # N x 1 x 1000 x 5
            mfb_iq_sumpool_t = torch.sum(mfb_iq_resh_t, 3, keepdim=True)    # N x 1 x 1000 x 1
            mfb_out_t = torch.squeeze(mfb_iq_sumpool_t)                     # N x 1000
            mfb_sign_sqrt_t = torch.sqrt(F.relu(mfb_out_t)) - torch.sqrt(F.relu(-mfb_out_t))
            mfb_l2_t = F.normalize(mfb_sign_sqrt_t)


            # aggregating function
            # calculating attention for each answer
            #ans_att = F.softmax(self.Dropout_C(self.multi_ans_attention(mfb_l2_i + mfb_l2_q + mfb_l2_t)))
            #out = torch.matmul(ans_att, torch.cat([mfb_l2_i.unsqueeze(1), mfb_l2_q.unsqueeze(1), mfb_l2_t.unsqueeze(1)],1))
            out = mfb_l2_t 
            #out = mfb_l2_t
            gate = torch.sigmoid(q_list[-1] * q_repr_q)
            out = gate * ans_list[-1] + (1-gate) * out

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels):

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
        return final_loss

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

def build_contextualized_reasoner_full(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    img_refiner = nn.Sequential(
        nn.Linear(img_embedding_size, img_embedding_size),
    )
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    updated_query_composer = FCNet([hidden_size + word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    neighbour_attention = MultiHeadedAttention(8, hidden_size, dropout=0.1)
    resize_ctx = weight_norm(nn.Linear(hidden_size + 512, 512))
    flattened_ctx_img = weight_norm(nn.Linear(512 * 7 * 7, hidden_size))
    Dropout_C = nn.Dropout(0.1)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Contextualized_Reasoner_Full(covnet, img_refiner, role_emb, verb_emb, query_composer, updated_query_composer, v_att, q_net,
                             v_net, neighbour_attention, resize_ctx, Dropout_C, flattened_ctx_img, classifier, encoder)


