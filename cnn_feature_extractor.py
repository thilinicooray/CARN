import torch
import json
import os
import h5py
import _pickle as cPickle

from sr import utils, imsitu_loader, imsitu_encoder
from sr.model import single_role_vgg_classifier


def extract_features(model, split, data_loader, gpu_mode, dataset_size):
    print('feature extraction started for split :', split)

    data_file_flat = {
        'train': 'data/verb_imsitu_train_flat.hdf5',
        'val': 'data/verb_imsitu_val_flat.hdf5',
        'test': 'data/verb_imsitu_test_flat.hdf5'}
    data_file_flat_relu = {
        'train': 'data/verb_imsitu_train_flat_relu.hdf5',
        'val': 'data/verb_imsitu_val_flat_relu.hdf5',
        'test': 'data/verb_imsitu_test_flat_relu.hdf5'}
    data_file_grid = {
        'train': 'data/verb_imsitu_train_grid.hdf5',
        'val': 'data/verb_imsitu_val_grid.hdf5',
        'test': 'data/verb_imsitu_test_grid.hdf5'}
    indices_file = {
        'train': 'data/verb_imsitu_train_imgid2idx.pkl',
        'val': 'data/verb_imsitu_val_imgid2idx.pkl',
        'test': 'data/verb_imsitu_test_imgid2idx.pkl'}
    ids_file = {
        'train': 'data/verb_imsitu_train_ids.pkl',
        'val': 'data/verb_imsitu_val_ids.pkl',
        'test': 'data/verb_imsitu_test_ids.pkl'}

    imgids = set()
    h_flat = h5py.File(data_file_flat[split], 'w')
    img_features_flat = h_flat.create_dataset(
        'image_features', (dataset_size, 4096), 'f')

    '''h_flat_relu = h5py.File(data_file_flat_relu[split], 'w')
    img_features_flat_relu = h_flat_relu.create_dataset(
        'image_features', (dataset_size, 1024), 'f')

    h_grid = h5py.File(data_file_grid[split], 'w')
    img_features_grid = h_grid.create_dataset(
        'image_features', (dataset_size, 1), 'f')'''

    counter = 0
    indices = {}
    model.eval()
    mx = len(data_loader)
    with torch.no_grad():
        for i, (img_id, img, labels) in enumerate(data_loader):
            print("{}/{} batches - {}\r".format(i+1,mx, split)),
            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
            else:
                img = torch.autograd.Variable(img)

            org_features = model.vgg_features(img)
            '''batch_size, n_channel, conv_h, conv_w = org_features.size()

            grid_features = org_features.view(batch_size, -1, conv_h* conv_w)
            grid_features = grid_features.permute(0, 2, 1)'''

            flat_features = model.classifier[:-1](org_features.view(-1, 512*7*7))
            #flat_features_relu = model.classifier[:-2](org_features.view(-1, 512*7*7))
            #pred_verb = torch.max(model.classifier(org_features.view(-1, 512*7*7)),-1)[1]

            batch_size = img.size(0)

            for j in range(batch_size):
                image_id = img_id[j]
                imgids.add(image_id)
                indices[image_id] = counter
                img_features_flat[counter, :] = flat_features[j].cpu().numpy()
                #img_features_flat_relu[counter, :] = flat_features_relu[j].cpu().numpy()
                #img_features_grid[counter] = pred_verb[j].cpu().numpy()
                counter += 1

    cPickle.dump(imgids, open(ids_file[split],'wb'))

    if counter != dataset_size:
        print('Missing images :', counter, len(imgids), dataset_size)

    cPickle.dump(indices, open(indices_file[split], 'wb'))
    h_flat.close()
    #h_flat_relu.close()
    #h_grid.close()
    print("done!")


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
    parser.add_argument('--model', type=str, default='single_role_classifier')
    parser.add_argument('--batch_size', type=int, default=64)
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

    train_set = imsitu_loader.imsitu_loader_verb(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    constructor = 'build_%s' % args.model
    model = getattr(single_role_vgg_classifier, constructor)(len(encoder.verb_list))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader.imsitu_loader_verb(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader_verb(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    if args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        args.train_all = True
        if len(args.resume_model) == 0:
            raise Exception('[pretrained module] not specified')
        utils.load_net(args.resume_model, [model])

    if args.gpuid >= 0:
        model.cuda()
    extract_features(model, 'train', train_loader, args.gpuid, len(train_loader)*batch_size)
    extract_features(model, 'val', dev_loader, args.gpuid, len(dev_loader)*batch_size)
    extract_features(model, 'test', test_loader, args.gpuid, len(test_loader)*batch_size)

if __name__ == "__main__":
    main()