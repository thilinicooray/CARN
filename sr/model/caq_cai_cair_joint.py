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


class Top_Down_Baseline(nn.Module):
    def __init__(self, caq_model, cai_model, flatten_img, reconstruct_img, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.caq_model = caq_model
        self.cai_model = cai_model
        self.flatten_img = flatten_img
        self.reconstruct_img = reconstruct_img
        self.encoder = encoder

    def forward(self, v_org, gt_verb):

        img_features = self.caq_model.convnet(v_org)
        flattened_img = self.flatten_img(img_features.view(-1, 512*7*7))

        caq_hidden_rep = self.caq_model.forward_hiddenrep(v_org, gt_verb)
        cai_hidden_rep = self.cai_model.forward_hiddenrep(v_org, gt_verb)

        joint_rep = caq_hidden_rep + cai_hidden_rep

        cur_group = joint_rep.contiguous().view(v_org.size(0), -1)
        constructed_img = self.reconstruct_img(cur_group)

        logits = self.caq_model.classifier(joint_rep)

        role_label_pred = logits.contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

        return role_label_pred, constructed_img, flattened_img

    def calculate_loss(self, gt_verbs, role_label_pred, gt_labels, constructed_img, flattened_img):

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.get_num_labels())
        l2_criterion = nn.MSELoss()

        final_loss_recons = 10*l2_criterion(constructed_img, flattened_img)

        gt_label_turned = gt_labels.transpose(1,2).contiguous().view(batch_size* self.encoder.max_role_count*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size* self.encoder.max_role_count, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3

        return loss + final_loss_recons

def build_caq_cai_joint(caq_model, cai_model, encoder):

    hidden_size = 1024

    flatten_img = nn.Sequential(
        nn.Linear(512*7*7, 1024),
        nn.BatchNorm1d(1024, momentum=0.01)
    )

    reconstruct_img = FCNet([hidden_size*6, hidden_size])

    return Top_Down_Baseline(caq_model, cai_model, flatten_img, reconstruct_img, encoder)