'''
Loading dataset for training and evaluation
'''

import torch.utils.data as data
from PIL import Image
import os
import random
import torch
import pickle as cPickle
import h5py
import numpy as np

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


class imsitu_loader_agent(data.Dataset):
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

        labels = self.encoder.encode_agent(ann)
        return _id, img, labels

    def __len__(self):
        return len(self.annotations)


class imsitu_loader_place(data.Dataset):
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

        labels = self.encoder.encode_place(ann)
        return _id, img, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb(data.Dataset):
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

        verb = self.encoder.encode_verb(ann)
        return _id, img, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_verb_pretrained_img_feat(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, split, transform=None, dataroot='data'):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

        #get verb grid features
        self.verb_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'verb_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading verb grid img features from h5 file')
        verb_grid_h5_path = os.path.join(dataroot, 'verb_imsitu_%s_grid.hdf5' % split)
        with h5py.File(verb_grid_h5_path, 'r') as hf:
            self.verb_features = np.array(hf.get('image_features'))

        self.verb_grid_features = torch.from_numpy(self.verb_features)

        #get agent flat features
        self.agent_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'agent_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading agent flat img features from h5 file')
        agent_flat_h5_path = os.path.join(dataroot, 'agent_imsitu_%s_flat_relu.hdf5' % split)
        with h5py.File(agent_flat_h5_path, 'r') as hf:
            self.agent_features = np.array(hf.get('image_features'))

        self.agent_flat_features = torch.from_numpy(self.agent_features)

        #get place flat features
        self.place_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'place_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading place flat img features from h5 file')
        place_flat_h5_path = os.path.join(dataroot, 'place_imsitu_%s_flat_relu.hdf5' % split)
        with h5py.File(place_flat_h5_path, 'r') as hf:
            self.place_features = np.array(hf.get('image_features'))

        self.place_flat_features = torch.from_numpy(self.place_features)

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        verb_grid_features = self.verb_grid_features[self.verb_img_id2idx[_id]]
        agent_flat_features = self.agent_flat_features[self.agent_img_id2idx[_id]]
        place_flat_features = self.place_flat_features[self.place_img_id2idx[_id]]



        verb = self.encoder.encode_verb(ann)
        return _id, verb_grid_features, agent_flat_features, place_flat_features, verb

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_top_down_verb(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, split, transform=None, dataroot='data'):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

        #get agent flat features
        self.agent_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'agent_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading agent flat img features from h5 file')
        agent_flat_h5_path = os.path.join(dataroot, 'agent_imsitu_%s_flat_relu.hdf5' % split)
        with h5py.File(agent_flat_h5_path, 'r') as hf:
            self.agent_features = np.array(hf.get('image_features'))

        self.agent_flat_features = torch.from_numpy(self.agent_features)

        #get place flat features
        self.place_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'place_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading place flat img features from h5 file')
        place_flat_h5_path = os.path.join(dataroot, 'place_imsitu_%s_flat_relu.hdf5' % split)
        with h5py.File(place_flat_h5_path, 'r') as hf:
            self.place_features = np.array(hf.get('image_features'))

        self.place_flat_features = torch.from_numpy(self.place_features)

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        agent_flat_features = self.agent_flat_features[self.agent_img_id2idx[_id]]
        place_flat_features = self.place_flat_features[self.place_img_id2idx[_id]]

        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)

        verb = self.encoder.encode_verb(ann)
        return _id, img, agent_flat_features, place_flat_features, verb

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

class imsitu_loader_verbimgfeat_4_role(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, split, transform=None, dataroot='data'):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

        #get verb grid features
        self.verb_img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'verb_imsitu_%s_imgid2idx.pkl' % split), 'rb'))
        print('loading verb grid img features from h5 file')
        verb_flat_h5_path = os.path.join(dataroot, 'verb_imsitu_%s_flat_relu.hdf5' % split)
        with h5py.File(verb_flat_h5_path, 'r') as hf:
            self.verb_features = np.array(hf.get('image_features'))

        self.verb_grid_features = torch.from_numpy(self.verb_features)

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        img = self.transform(img)
        img_feat = self.verb_grid_features[self.verb_img_id2idx[_id]]

        verb, labels = self.encoder.encode(ann)
        return _id, img, img_feat, verb, labels

    def __len__(self):
        return len(self.annotations)