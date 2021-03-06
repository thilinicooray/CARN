'''
reasoner - top-down baseline
fusioner - pairwise only
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.nn.utils.weight_norm import weight_norm

from ..lib.attention import Attention
from ..lib.classifier import SimpleClassifier
from ..lib.fc import FCNet
from ..utils import cross_entropy_loss

class vgg16_modified(nn.Module):
    def __init__(self):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features

    def forward(self,x):
        features = self.vgg_features(x)
        return features

class Top_Down_With_Pair_Rf(nn.Module):
    def __init__(self, convnet, role_emb, verb_emb, query_composer, v_att, q_net, v_net, neighbour_att, classifier, encoder, Dropout_C):
        super(Top_Down_With_Pair_Rf, self).__init__()
        self.convnet = convnet
        self.role_emb = role_emb
        self.verb_emb = verb_emb
        self.query_composer = query_composer
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.neighbour_att = neighbour_att
        self.classifier = classifier
        self.encoder = encoder
        self.dropout = Dropout_C

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
        concat_query = torch.cat([ verb_embed_expand, role_embd], -1)
        role_verb_embd = concat_query.contiguous().view(-1, role_embd.size(-1)*2)
        q_emb = self.query_composer(role_verb_embd)

        att = self.v_att(img, q_emb)
        v_emb = (att * img).sum(1)
        v_repr = self.v_net(v_emb)
        q_repr = self.q_net(q_emb)

        out = q_repr * v_repr

        #normalization as a data range of a multiplication can be really large
        '''iq_drop = self.dropout(out)
        iq_sign_sqrt = torch.sqrt(F.relu(iq_drop)) - torch.sqrt(F.relu(-iq_drop))
        iq_l2 = F.normalize(iq_sign_sqrt)'''
        #print('out normalized ', iq_l2[0,:10])

        #pairwise context generator
        all_roles = out.contiguous().view(v.size(0), self.encoder.max_role_count, -1)

        required_indices = [[1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]]

        updated_roles = None

        for rolei in range(self.encoder.max_role_count):
            current_indices = required_indices[rolei]
            current_indices = torch.tensor(current_indices)
            if torch.cuda.is_available():
                current_indices = current_indices.to(torch.device('cuda'))

            neighbours = torch.index_select(all_roles, 1, current_indices)

            current_role = all_roles[:,rolei]

            neighbour_att_weights = self.neighbour_att(neighbours, current_role)
            context = (neighbour_att_weights * neighbours).sum(1)



            if rolei == 0:
                updated_roles = joint.unsqueeze(1)
            else:
                updated_roles = torch.cat((updated_roles.clone(), joint.unsqueeze(1)), 1)

        final_out = updated_roles.contiguous().view(v.size(0)* self.encoder.max_role_count, -1)
        logits = self.classifier(final_out)

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

def build_top_down_with_pair_rf(n_roles, n_verbs, num_ans_classes, encoder):

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

    neighbour_att = Attention(hidden_size, hidden_size, hidden_size)

    Dropout_C = nn.Dropout(0.2)

    classifier = SimpleClassifier(
        hidden_size, 2 * hidden_size, num_ans_classes, 0.5)

    return Top_Down_With_Pair_Rf(covnet, role_emb, verb_emb, query_composer, v_att, q_net,
                                 v_net, neighbour_att, classifier, encoder, Dropout_C)


