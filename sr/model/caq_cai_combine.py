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
    def __init__(self, caq_model, cai_model, encoder):
        super(Top_Down_Baseline, self).__init__()
        self.caq_model = caq_model
        self.cai_model = cai_model
        self.encoder = encoder

    def forward(self, v_org, gt_verb):

        caq_hidden_rep = self.caq_model.forward_hiddenrep(v_org, gt_verb)
        cai_hidden_rep = self.cai_model.forward_hiddenrep(v_org, gt_verb)

        final_rep = caq_hidden_rep + cai_hidden_rep

        logits = self.caq_model.classifier(final_rep)

        role_label_pred = logits.contiguous().view(v_org.size(0), self.encoder.max_role_count, -1)

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

def build_caq_cai_joint(caq_model, cai_model, encoder):

    return Top_Down_Baseline(caq_model, cai_model, encoder)