'''
We try to recreate GGNN based SR model in PyTorch this class
'''

import torch
import torch.nn as nn
import torchvision as tv
from ..utils import cross_entropy_loss

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


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim, n_node,  n_steps):
        super(GGNN, self).__init__()

        self.state_dim = state_dim
        self.n_node = n_node
        self.n_steps = n_steps

        #neighbour projection
        self.W_p = nn.Linear(state_dim, state_dim)
        #weights of update gate
        self.W_z = nn.Linear(state_dim, state_dim)
        self.U_z = nn.Linear(state_dim, state_dim)
        #weights of reset gate
        self.W_r = nn.Linear(state_dim, state_dim)
        self.U_r = nn.Linear(state_dim, state_dim)
        #weights of transform
        self.W_h = nn.Linear(state_dim, state_dim)
        self.U_h = nn.Linear(state_dim, state_dim)

    def forward(self, init_node, mask):

        hidden_state = init_node

        for t in range(self.n_steps):
            # calculating neighbour info
            neighbours = hidden_state.contiguous().view(mask.size(0), self.n_node, -1)
            print('neighbours first', neighbours.size(), mask.size())
            neighbours = neighbours.expand(self.n_node, neighbours.size(0), neighbours.size(1), neighbours.size(2))
            neighbours = neighbours.transpose(0,1)
            print('neighbours second', neighbours.size(), neighbours[0,0,:,:5], mask[0,0])

            neighbours = neighbours * mask.unsqueeze(-1)
            print('masked neighbours ', neighbours[0,0,:,:5])
            neighbours = self.W_p(neighbours)
            neighbours = torch.sum(neighbours, 2)
            print('masked summed neighbours ', neighbours.size())
            neighbours = neighbours.contiguous().view(mask.size(0)*self.n_node, -1)
            print('neighbours + self ', neighbours.size(), hidden_state.size())

            #applying gating
            z_t = torch.sigmoid(self.W_z(neighbours) + self.U_z(hidden_state))
            r_t = torch.sigmoid(self.W_r(neighbours) + self.U_r(hidden_state))
            h_hat_t = torch.tanh(self.W_h(neighbours) + self.U_h(r_t * hidden_state))
            hidden_state = (1 - z_t) * hidden_state + z_t * h_hat_t

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

        mask = self.encoder.get_adj_matrix_noself(gt_verb)
        if torch.cuda.is_available():
            mask = mask.to(torch.device('cuda'))

        out = self.ggnn(input2ggnn, mask)

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

def build_ggnn_baseline(n_roles, n_verbs, num_ans_classes, encoder):

    hidden_size = 1024

    covnet = vgg16_modified()
    role_emb = nn.Embedding(n_roles+1, hidden_size, padding_idx=n_roles)
    verb_emb = nn.Embedding(n_verbs, hidden_size)
    ggnn = GGNN(state_dim = hidden_size, n_node=encoder.max_role_count,
                n_steps=4)
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(hidden_size, num_ans_classes)
    )

    return GGNN_Baseline(covnet, role_emb, verb_emb, ggnn, classifier, encoder)