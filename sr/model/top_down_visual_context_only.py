'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm

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

class Top_Down_Baseline(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, resize_ctx, Dropout_C, classifier, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.resize_ctx = resize_ctx
        self.Dropout_C = Dropout_C
        self.classifier = classifier
        self.encoder = encoder

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

    def forward(self, v_org, gt_verb):

        v_list = []
        ans_list = []

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

        out = q_repr * v_repr

        v_list.append(v_repr)
        ans_list.append(out)

        for i in range(1):

            cur_group = out.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

            neighbours = self.attention(cur_group, cur_group, cur_group, mask=mask)

            withctx = neighbours.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)

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
            updated_img = added_img * img

            att = self.v_att(updated_img, q_emb)
            v_emb = (att * updated_img).sum(1)
            v_repr = self.v_net(v_emb)
            q_repr = self.q_net(q_emb)

            out = q_repr * v_repr

            gate = torch.sigmoid(v_list[-1] * v_repr)
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

def build_top_down_visual_context_only_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = FCNet([word_embedding_size * 2, hidden_size])
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])

    resize_ctx = weight_norm(nn.Linear(hidden_size + 512, 512))
    Dropout_C = nn.Dropout(0.1)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Baseline(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, resize_ctx, Dropout_C, classifier, encoder)


