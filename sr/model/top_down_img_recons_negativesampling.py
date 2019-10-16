'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, Dropout_C, flatten_img, classifier, encoder, reconstruct_img):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.Dropout_C = Dropout_C
        self.flatten_img = flatten_img
        self.classifier = classifier
        self.encoder = encoder
        self.reconstruct_img = reconstruct_img

    def forward(self, v_org, gt_verb, negative_samples):
        print('org size ', v_org.size(), negative_samples.size())
        n_heads = 1
        img_features = self.convnet(v_org)

        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

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

        mask = self.encoder.get_adj_matrix(gt_verb)

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


        cur_group = out.contiguous().view(v.size(0), -1)

        if self.training:
            flattened_img = self.flatten_img(img_features.view(-1, 512*7*7))
            flattened_img = flattened_img.expand(5, flattened_img.size(0), flattened_img.size(1))
            flattened_img = flattened_img.view(batch_size* 5, -1)

            negative_img_all = negative_samples.view(batch_size*5, 4)
            negative_img_embed = self.flatten_img(self.convnet(negative_img_all).view(-1, 512*7*7))
            #unmasked encoding
            constructed_img = self.reconstruct_img(cur_group)
            constructed_img = constructed_img.expand(5, constructed_img.size(0), constructed_img.size(1))
            constructed_img = constructed_img.view(batch_size* 5, -1)

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        if self.training:
            return role_label_pred, constructed_img, flattened_img, negative_img_embed

        else:
            return role_label_pred

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels, constructed_img, flattened_img, negative_img_embed):

        #l2_criterion = nn.MSELoss()

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                for j in range(0, self.encoder.max_role_count):
                    frame_loss += cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.encoder.get_num_labels())
                frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                loss += frame_loss

        final_loss_entropy = loss/batch_size

        #margin ranking loss
        margin = torch.bmm(negative_img_embed.view(negative_img_embed.size(0), 1, negative_img_embed.size(1))
                           , flattened_img.view(flattened_img.size(0), flattened_img.size(1), 1))

        value = torch.bmm(constructed_img.view(negative_img_embed.size(0), 1, negative_img_embed.size(1))
                          , (flattened_img - negative_img_embed).view(flattened_img.size(0), flattened_img.size(1), 1))

        print('rank loss ', value.size(), margin.size() )

        rank_loss = torch.sum(torch.max(0, margin - value),0)/batch_size

        return final_loss_entropy, rank_loss


def build_top_down_img_recons(n_roles, n_verbs, num_ans_classes, encoder):

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
    Dropout_C = nn.Dropout(0.1)

    flatten_img = nn.Sequential(
        nn.Linear(512*7*7, 1024),
        nn.BatchNorm1d(1024, momentum=0.01)
    )

    reconstruct_img = FCNet([hidden_size*6, hidden_size])

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_Baseline(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, Dropout_C, flatten_img, classifier, encoder, reconstruct_img)


