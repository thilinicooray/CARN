import torch
import json
import os
import time

from sr import utils, imsitu_scorer, imsitu_scorer_rare, imsitu_loader, imsitu_encoder
from sr.model import top_down_baseline
from sr.tools.td_feature_extractor import extract_features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, model_name, model_saving_name, eval_frequency=4000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []

    if gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model
    #pmodel = model

    all = count_parameters(model)
    cnn = count_parameters(model.convnet)

    print('model parameters - all, cnn, base ', all, cnn)

    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

    for epoch in range(max_epoch):
        t = time.time()
        for i, (_, img, verb, labels) in enumerate(train_loader):
            total_steps += 1

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            role_predict = pmodel(img, verb)
            loss = model.calculate_loss(verb, role_predict, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            top1.add_point_noun(verb, role_predict, labels)
            top5.add_point_noun(verb, role_predict, labels)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results_nouns()
                top5_a = top5.get_average_results_nouns()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results_nouns()
                top5_avg = top5.get_average_results_nouns()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))
                dev_score_list.append(avg_score)
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    torch.save(model.state_dict(), model_dir + "/{}_{}.model".format( model_name, model_saving_name))
                    print ('New best model saved! {0}'.format(max_score))

                print('current train loss', train_loss)
                train_loss = 0
                top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

            del role_predict, loss, img, verb, labels
        print('epoch %d, time: %.2f' % (epoch, time.time()-t))
        print('Epoch ', epoch, ' completed!')
        scheduler.step()

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        t1 = time.time()
        for i, (img_id, img, verb, labels) in enumerate(dev_loader):

            #print(img_id[0], encoder.verb2_role_dict[encoder.verb_list[verb[0]]])

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            role_predict = model(img, verb)

            if write_to_file:
                top1.add_point_noun_log(img_id, verb, role_predict, labels)
                top5.add_point_noun_log(img_id, verb, role_predict, labels)
            else:
                top1.add_point_noun(verb, role_predict, labels)
                top5.add_point_noun(verb, role_predict, labels)

            del role_predict, img, verb, labels
            #break
        print('eval, time: %.2f' % ( time.time()-t1))
    return top1, top5, 0

def eval_output(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    img_id_list = ['vaulting_90.jpg', 'dialing_213.jpg', 'stirring_68.jpg', 'ailing_175.jpg', 'unloading_97.jpg',
                   'clenching_115.jpg', 'imitating_44.jpg', 'assembling_230.jpg', 'wrapping_101.jpg', 'descending_105.jpg',
                   'stuffing_202.jpg', 'dining_67.jpg', 'tattooing_72.jpg', 'fetching_83.jpg', 'giving_46.jpg',
                   'striking_202.jpg', 'tickling_86.jpg', 'brewing_110.jpg', 'taping_142.jpg', 'fording_40.jpg']

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():

        for i, (img_id, img, verb, labels) in enumerate(dev_loader):

            #print(img_id[0], encoder.verb2_role_dict[encoder.verb_list[verb[0]]])
            show_att = False
            if img_id[0] in img_id_list:
                print('handling ', img_id[0])
                show_att = True
            else:
                continue

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            role_predict = model.forward_vis(img, verb, show_att)

            if write_to_file:
                top1.add_point_noun_log(img_id, verb, role_predict, labels)
                top5.add_point_noun_log(img_id, verb, role_predict, labels)
            else:
                top1.add_point_noun(verb, role_predict, labels)
                top5.add_point_noun(verb, role_predict, labels)

            del role_predict, img, verb, labels
            #break

    return top1, top5, 0

def eval_rare(model, dev_loader, encoder, gpu_mode, image_group = {}):
    model.eval()

    print ('evaluating rare portion ...')
    top1 = imsitu_scorer_rare.imsitu_scorer(encoder, 1, 3, image_group)
    top5 = imsitu_scorer_rare.imsitu_scorer(encoder, 5, 3, image_group)
    with torch.no_grad():

        for i, (img_id, img, verb, labels) in enumerate(dev_loader):

            #print(img_id[0], encoder.verb2_role_dict[encoder.verb_list[verb[0]]])

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)

            role_predict = model(img, verb)

            top1.add_point_noun_log(img_id, verb, role_predict, labels)
            top5.add_point_noun_log(img_id, verb, role_predict, labels)

            del role_predict, img, verb, labels
            #break

    return top1, top5, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--evaluate_visualize', action='store_true', help='Only use the testing mode to visualize ')
    parser.add_argument('--evaluate_rare', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
    parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
    parser.add_argument('--org_train_file', default="train.json", type=str, help='org train file name')
    parser.add_argument('--org_test_file', default="test.json", type=str, help='org test file name')
    parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default='top_down_baseline')
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

    train_set = imsitu_loader.imsitu_loader(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    constructor = 'build_%s' % args.model
    model = getattr(top_down_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader.imsitu_loader(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
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
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3)
        model_name = 'resume_all'


    if args.gpuid >= 0:
        model.cuda()
    extract_features(model, 'train', train_loader, args.gpuid, len(train_loader)*batch_size)
    extract_features(model, 'val', dev_loader, args.gpuid, len(dev_loader)*batch_size)
    extract_features(model, 'test', test_loader, args.gpuid, len(test_loader)*batch_size)





if __name__ == "__main__":
    main()












