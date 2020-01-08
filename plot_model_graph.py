import torch
import torch.nn as nn

import torchvision.models as models

from torchviz import make_dot
from collections import OrderedDict

import torch
import json
import os

from sr import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from sr.model import adaptive_base_qctx_e2e


def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, model_name, model_saving_name, eval_frequency=4000):

    for i, (_, img, verb, labels) in enumerate(train_loader):

        img = torch.autograd.Variable(img.cuda())
        verb = torch.autograd.Variable(verb.cuda())
        labels = torch.autograd.Variable(labels.cuda())

        y = model(img, verb)
        dot = make_dot(y, params=OrderedDict((name, param) for (name, param) in model.named_parameters()))

        dot.format = 'pdf'
        dot.render('adaptive_base_qctx_e2e')

        break


def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
    parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
    parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default='top_down_query_context_only_baseline')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=3)

    args = parser.parse_args()

    n_epoch = args.epochs
    batch_size = args.batch_size
    clip_norm = args.clip_norm
    n_worker = args.num_workers

    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    train_set = json.load(open(dataset_folder + '/' + args.train_file))

    encoder = imsitu_encoder.imsitu_encoder(train_set)

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    constructor = 'build_%s' % args.model
    model = getattr(adaptive_base_qctx_e2e, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)


    train(model, train_loader, None, None, None, n_epoch, args.output_dir, encoder, args.gpuid, clip_norm, None, args.model_saving_name,
              )






if __name__ == "__main__":
    main()












