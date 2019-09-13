'''
Loading dataset for training and evaluation
'''

import torch.utils.data as data
from PIL import Image
import os

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

        return _id, img, verb, labels

    def __len__(self):
        return len(self.annotations)