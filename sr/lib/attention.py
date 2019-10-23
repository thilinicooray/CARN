import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from ..lib.fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)

        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class Context_Erased_Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Context_Erased_Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q, mask):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = logits
        single_att = nn.functional.softmax(logits, 1)

        w = w.contiguous().view(mask.size(0), mask.size(1), -1)
        w_ctx = w * mask.unsqueeze(-1)

        required_indices = [[1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]]

        sum_att = torch.ones(mask.size(0), w.size(-1))
        if torch.cuda.is_available():
            sum_att = sum_att.to(torch.device('cuda'))

        updated_att = None

        for rolei in range(w.size(1)):
            current_indices = required_indices[rolei]
            current_indices = torch.tensor(current_indices)
            if torch.cuda.is_available():
                current_indices = current_indices.to(torch.device('cuda'))

            neighbour_att = torch.sum(torch.index_select(w_ctx, 1, current_indices),1)
            neighbour_removed_att = sum_att - neighbour_att

            #updated_cur_role_att = nn.functional.softmax(neighbour_removed_att + w[:,rolei],1)
            updated_cur_role_att = nn.functional.softmax(neighbour_removed_att ,1)

            if rolei == 0:
                updated_att = updated_cur_role_att.unsqueeze(1)
            else:
                updated_att = torch.cat((updated_att.clone(), updated_cur_role_att.unsqueeze(1)), 1)


        context_erased_att = updated_att.contiguous().view(-1, updated_att.size(-1), 1)

        return single_att, context_erased_att

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits
