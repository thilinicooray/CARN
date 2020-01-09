'''
This is the baseline model.
We directly use bottom up VQA like mechanism for SR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..lib.attention import Context_Erased_Attention
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
    def __init__(self, covnet, baseline_model, encoder, Dropout_C, obj_cls):
        super(Top_Down_Baseline, self).__init__()
        self.convnet = covnet
        self.baseline_model = baseline_model
        self.encoder = encoder
        self.Dropout_C = Dropout_C
        self.obj_cls = obj_cls

    def forward(self, v_org, gt_verb):

        role_oh_encoding = self.encoder.get_verb2role_encoing_batch(gt_verb)
        if torch.cuda.is_available():
            role_oh_encoding = role_oh_encoding.to(torch.device('cuda'))

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

        verb_embd = self.baseline_model.verb_emb(gt_verb)
        role_embd = self.baseline_model.role_emb(role_idx)

        verb_embed_expand = verb_embd.expand(self.encoder.max_role_count, verb_embd.size(0), verb_embd.size(1))
        verb_embed_expand = verb_embed_expand.transpose(0,1)
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.baseline_model.query_composer(role_verb_embd)

        att, context_erased_att = self.baseline_model.v_att.forward_with_ctxatt_mask(img, q_emb, role_oh_encoding)
        ctx_erased_v_emb = (context_erased_att * img).sum(1)

        logits_obj = self.obj_cls(ctx_erased_v_emb)

        logits = logits_obj

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

def build_top_down_baseline(n_roles, n_verbs, num_ans_classes, encoder, baseline_model):

    hidden_size = 1024
    img_embedding_size = 512
    covnet = vgg16_modified()

    obj_cls = nn.Sequential(
        nn.Linear(img_embedding_size, hidden_size*2),
        nn.BatchNorm1d(hidden_size*2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size*2, num_ans_classes)
    )

    Dropout_C = nn.Dropout(0.1)

    return Top_Down_Baseline(covnet, baseline_model, encoder, Dropout_C, obj_cls)


