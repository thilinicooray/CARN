'''
Loading dataset for training and evaluation
'''

import torch.utils.data as data
from PIL import Image
import os
import random
import torch

class imsitu_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        verb, labels = self.encoder.encode(ann)
        print(_id, ann)
        return _id, img, verb, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_negative_sampling(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, dictionary, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.max_id = len(self.ids) - 1
        self.encoder = encoder
        self.dictionary = dictionary
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        #getting negative samples
        sample_count = 1
        sample_idx_list = []
        sample_list = []
        is_incomplete = True
        while is_incomplete:
            idx = random.randint(0, self.max_id)
            if idx not in sample_idx_list and idx != index:
                _idx = self.ids[idx]
                neg_img = Image.open(os.path.join(self.img_dir, _idx)).convert('RGB')
                neg_img = self.transform(neg_img)

                sample_idx_list.append(_idx)
                sample_list.append(neg_img)

                if len(sample_idx_list) == sample_count:
                    is_incomplete = False

        verb, labels = self.encoder.encode(ann)

        return _id, img, verb, labels, torch.stack(sample_list,0)

    def __len__(self):
        return len(self.annotations)