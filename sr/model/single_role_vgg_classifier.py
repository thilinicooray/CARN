'''
This is the CNN classifier for agent and place roles to get the context for other queries.
'''

import torch
import torch.nn as nn

import torchvision as tv
from ..utils import cross_entropy_loss

class vgg16_modified(nn.Module):
    def __init__(self, num_classes):
        super(vgg16_modified, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        self.vgg_features = vgg.features
        self.num_ans_classes = num_classes

        '''num_features = vgg.classifier[3].in_features
        features = list(vgg.classifier.children())[:-4]

        new_classifier = nn.Sequential(
            nn.Linear(num_features, num_features//4),
            #nn.BatchNorm1d(num_features//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_features//4, num_classes)
        )

        features.extend(new_classifier)
        self.classifier = nn.Sequential(*features)'''

        num_features = vgg.classifier[6].in_features
        features = list(vgg.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(num_classes))]) # Add our layer with 4 outputs
        self.classifier = nn.Sequential(*features)


    def forward(self,x):
        features = self.vgg_features(x)
        out = self.classifier(features.view(-1, 512*7*7))

        return out

    '''def calculate_loss(self, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]

        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                loss += cross_entropy_loss(role_label_pred[i], gt_labels[i,index], self.num_ans_classes)

        final_loss = loss/batch_size
        return final_loss'''

    def calculate_loss(self, role_label_pred, gt_labels):

        batch_size = role_label_pred.size()[0]
        criterion = nn.CrossEntropyLoss()

        gt_label_turned = gt_labels.contiguous().view(batch_size*3, -1)

        role_label_pred = role_label_pred.contiguous().view(batch_size, -1)
        role_label_pred = role_label_pred.expand(3, role_label_pred.size(0), role_label_pred.size(1))
        role_label_pred = role_label_pred.transpose(0,1)
        role_label_pred = role_label_pred.contiguous().view(-1, role_label_pred.size(-1))

        loss = criterion(role_label_pred, gt_label_turned.squeeze(1)) * 3

        return loss

    def calculate_verb_loss(self, verb_pred, gt_verbs):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = 0
            verb_loss += cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss


        final_loss = loss/batch_size
        return final_loss

def build_single_role_classifier(num_ans_classes):

    covnet = vgg16_modified(num_ans_classes)

    return covnet


