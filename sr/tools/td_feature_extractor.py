import os
import argparse
import sys

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import torch


def extract_features(model, split, data_loader, gpu_mode, dataset_size):
    print('feature extraction started for split :', split)

    data_file = {
        'train': 'data/td_imsitu_train.hdf5',
        'val': 'data/td_imsitu_val.hdf5',
        'test': 'data/td_imsitu_test.hdf5'}
    indices_file = {
        'train': 'data/td_imsitu_train_imgid2idx.pkl',
        'val': 'data/td_imsitu_val_imgid2idx.pkl',
        'test': 'data/td_imsitu_test_imgid2idx.pkl'}
    ids_file = {
        'train': 'data/td_imsitu_train_ids.pkl',
        'val': 'data/td_imsitu_val_ids.pkl',
        'test': 'data/td_imsitu_test_ids.pkl'}

    feature_length = 1024
    imgids = set()
    h = h5py.File(data_file[split], 'w')
    img_features = h.create_dataset(
        'td_features', (dataset_size, 6, feature_length), 'f')

    counter = 0
    indices = {}
    model.eval()
    mx = len(data_loader)
    with torch.no_grad():
        for i, (img_id, img, verb, labels) in enumerate(data_loader):
            print("{}/{} batches - {}\r".format(i+1,mx, split)),
            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            features = model.forward_hiddenrep(img, verb)


            batch_size = img.size(0)
            features = features.contiguous().view(batch_size, 6, -1)

            for j in range(batch_size):
                image_id = img_id[j]
                imgids.add(image_id)
                indices[image_id] = counter
                img_features[counter, :, :] = features[j].cpu().numpy()
                counter += 1

    cPickle.dump(imgids, open(ids_file[split],'wb'))

    if counter != dataset_size:
        print('Missing images :', counter, len(imgids), dataset_size)

    cPickle.dump(indices, open(indices_file[split], 'wb'))
    h.close()
    print("done!")


