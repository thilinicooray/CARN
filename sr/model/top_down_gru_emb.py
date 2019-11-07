'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, classifier, encoder, Dropout_C):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.encoder = encoder
        self.Dropout_C = Dropout_C

    def forward(self, v_org, gt_verb):

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
        concat_query = torch.cat([verb_embed_expand.unsqueeze(-2), role_embd.unsqueeze(-2)],-2)
        role_verb_embd = concat_query.contiguous().view(-1, concat_query.size(-2), role_embd.size(-1))
        self.query_composer.flatten_parameters()
        output, hidden = self.query_composer(role_verb_embd)

        q_emb = output[:, -1]
        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        '''print(' analysis ')
        att1 = att.contiguous().view(batch_size,6, 7,7)

        print(att1[0])'''

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* self.encoder.max_role_count, 1, -1, 1)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        return role_label_pred

    def forward_agentplace_noverb(self, v_org):

        max_role_count = 2

        img_features = self.convnet(v_org)
        batch_size, n_channel, conv_h, conv_w = img_features.size()

        img_org = img_features.view(batch_size, -1, conv_h* conv_w)
        v = img_org.permute(0, 2, 1)

        batch_size = v.size(0)

        role_idx = self.encoder.get_agent_place_ids_batch(batch_size)

        if torch.cuda.is_available():
            role_idx = role_idx.to(torch.device('cuda'))

        img = v

        img = img.expand(max_role_count, img.size(0), img.size(1), img.size(2))

        img = img.transpose(0,1)
        img = img.contiguous().view(batch_size * max_role_count, -1, v.size(2))

        #verb_embd = torch.sum(self.verb_emb.weight, 0)
        #verb_embd = verb_embd.expand(batch_size, verb_embd.size(-1))
        #verb_embd = torch.zeros(batch_size, 300).cuda()
        role_embd = self.role_emb(role_idx)

        concat_query = role_embd
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1))
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)

        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        out = q_repr * v_repr

        '''mfb_iq_eltwise = torch.mul(q_repr, v_repr)

        mfb_iq_drop = self.Dropout_C(mfb_iq_eltwise)

        mfb_iq_resh = mfb_iq_drop.view(batch_size* max_role_count, 1, -1, 1)   # N x 1 x 1000 x 5
        mfb_iq_sumpool = torch.sum(mfb_iq_resh, 3, keepdim=True)    # N x 1 x 1000 x 1
        mfb_out = torch.squeeze(mfb_iq_sumpool)                     # N x 1000
        mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        mfb_l2 = F.normalize(mfb_sign_sqrt)
        out = mfb_l2'''

        logits = self.classifier(out)

        role_label_pred = logits.contiguous().view(v.size(0), max_role_count, -1)
        role_label_rep = out.contiguous().view(v.size(0), max_role_count, -1)

        return role_label_pred, role_label_rep

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

def build_top_down_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024
    word_embedding_size = 300
    img_embedding_size = 512

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, word_embedding_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, word_embedding_size)
    query_composer = nn.GRU(
        word_embedding_size, hidden_size, 1,
        bidirectional=False,
        dropout=0.0,
        batch_first=True)
    v_att = Attention(img_embedding_size, hidden_size, hidden_size)
    q_net = FCNet([hidden_size, hidden_size ])
    v_net = FCNet([img_embedding_size, hidden_size])
    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes + 1, 0.5)

    Dropout_C = nn.Dropout(0.1)

    return Top_Down_Baseline(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                             v_net, classifier, encoder, Dropout_C)


