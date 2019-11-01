import torch
import json
import os

from sr import utils, imsitu_scorer, imsitu_loader, imsitu_encoder
from sr.model import top_down_verb_full, single_role_vgg_classifier


def train(model, train_loader, dev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, model_name, model_saving_name, eval_frequency=4000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []

    '''if gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model'''
    pmodel = model

    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)

    for epoch in range(max_epoch):

        for i, (img_id, img, agent_flat_features, place_flat_features, verb) in enumerate(train_loader):
            total_steps += 1

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                agent_flat_features = torch.autograd.Variable(agent_flat_features.cuda())
                place_flat_features = torch.autograd.Variable(place_flat_features.cuda())
                verb = torch.autograd.Variable(verb.cuda())
            else:
                img = torch.autograd.Variable(img)
                agent_flat_features = torch.autograd.Variable(agent_flat_features)
                place_flat_features = torch.autograd.Variable(place_flat_features)
                verb = torch.autograd.Variable(verb)

            verb_predict = pmodel(img, agent_flat_features, place_flat_features)
            loss = model.calculate_verb_loss(verb_predict, verb)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            top1.add_point_verb_only_eval(img_id, verb_predict, verb)
            top5.add_point_verb_only_eval(img_id, verb_predict, verb)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

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

            del verb_predict, loss, img, agent_flat_features, place_flat_features, verb
        print('Epoch ', epoch, ' completed!')
        scheduler.step()

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():

        for i, (img_id, img, agent_flat_features, place_flat_features, verb) in enumerate(dev_loader):

            #print(img_id[0], encoder.verb2_role_dict[encoder.verb_list[verb[0]]])

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                agent_flat_features = torch.autograd.Variable(agent_flat_features.cuda())
                place_flat_features = torch.autograd.Variable(place_flat_features.cuda())
                verb = torch.autograd.Variable(verb.cuda())
            else:
                img = torch.autograd.Variable(img)
                agent_flat_features = torch.autograd.Variable(agent_flat_features)
                place_flat_features = torch.autograd.Variable(place_flat_features)
                verb = torch.autograd.Variable(verb)

            verb_predict = model(img, agent_flat_features, place_flat_features)

            top1.add_point_verb_only_eval(img_id, verb_predict, verb)
            top5.add_point_verb_only_eval(img_id, verb_predict, verb)

            del verb_predict, img, agent_flat_features, place_flat_features, verb
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
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--train_file', default="train_freq2000.json", type=str, help='trainfile name')
    parser.add_argument('--dev_file', default="dev_freq2000.json", type=str, help='dev file name')
    parser.add_argument('--test_file', default="test_freq2000.json", type=str, help='test file name')
    parser.add_argument('--model_saving_name', type=str, help='saving name of the outpul model')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default='top_down_baseline_verb')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=3)

    parser.add_argument('--agent_model', type=str, default='', help='Pretrained agent classifier')
    parser.add_argument('--place_model', type=str, default='', help='Pretrained place classifier')

    args = parser.parse_args()

    n_epoch = args.epochs
    batch_size = args.batch_size
    clip_norm = args.clip_norm
    n_worker = args.num_workers

    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    train_set = json.load(open(dataset_folder + '/' + args.train_file))

    encoder = imsitu_encoder.imsitu_encoder(train_set)

    train_set = imsitu_loader.imsitu_loader_top_down_verb(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    #loading classifier part from pretrained agent and place models
    constructor = 'build_single_role_classifier'
    agent_model = getattr(single_role_vgg_classifier, constructor)(len(encoder.agent_label_list))

    utils.load_net(args.agent_model, [agent_model])

    place_model = getattr(single_role_vgg_classifier, constructor)(len(encoder.place_label_list))
    utils.load_net(args.place_model, [place_model])

    #building current model
    constructor = 'build_%s' % args.model
    model = getattr(top_down_verb_full, constructor)(len(encoder.agent_label_list), len(encoder.place_label_list),
                                                     agent_model.classifier[-3:], place_model.classifier[-3:], len(encoder.verb_list))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader.imsitu_loader_top_down_verb(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader_top_down_verb(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
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

    else:
        print('Training from the scratch.')
        model_name = 'train_full'
        utils.set_trainable(model, True)
        utils.set_trainable(model.agent_classifier, False)
        utils.set_trainable(model.place_classifier, False)
        optimizer = torch.optim.Adamax([
            {'params': model.convnet.parameters(), 'lr': 5e-5},
            {'params': model.conv_exp.parameters()},
            {'params': model.agent_emb.parameters()},
            {'params': model.place_emb.parameters()},
            {'params': model.resize_img_flat.parameters()},
            {'params': model.resize_img_grid.parameters()},
            {'params': model.query_composer.parameters()},
            {'params': model.v_att.parameters()},
            {'params': model.q_net.parameters()},
            {'params': model.v_net.parameters()},
            {'params': model.classifier.parameters()}
        ], lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if args.evaluate:
        top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.2f}', '5-')))

        #write results to csv file
        role_dict = top1.role_dict
        fail_val_all = top1.value_all_dict
        pass_val_dict = top1.vall_all_correct

        with open(args.model_saving_name+'_role_pred_data.json', 'w') as fp:
            json.dump(role_dict, fp, indent=4)

        with open(args.model_saving_name+'_fail_val_all.json', 'w') as fp:
            json.dump(fail_val_all, fp, indent=4)

        with open(args.model_saving_name+'_pass_val_all.json', 'w') as fp:
            json.dump(pass_val_dict, fp, indent=4)

        print('Writing predictions to file completed !')

    elif args.test:
        top1, top5, val_loss = eval(model, test_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results_nouns()
        top5_avg = top5.get_average_results_nouns()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                    utils.format_dict(top5_avg, '{:.2f}', '5-')))


    else:

        print('Model training started!')
        train(model, train_loader, dev_loader, optimizer, scheduler, n_epoch, args.output_dir, encoder, args.gpuid, clip_norm, model_name, args.model_saving_name,
              )






if __name__ == "__main__":
    main()












