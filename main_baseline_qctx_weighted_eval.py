import torch
import json
import os

from sr import utils, imsitu_scorer, imsitu_scorer_rare, imsitu_loader, imsitu_encoder
from sr.model import baseline_qctx_weighted_joint_eval, top_down_query_context_pretrained_baseline, top_down_baseline




def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()

    print ('evaluating model...')
    top1 = imsitu_scorer.imsitu_scorer(encoder, 1, 3, write_to_file)
    top5 = imsitu_scorer.imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img_id, img, verb, labels, verb_role_impact) in enumerate(dev_loader):
            #print("{}/{} batches\r".format(i+1,mx))
            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
                verb_role_impact = torch.autograd.Variable(verb_role_impact.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                labels = torch.autograd.Variable(labels)
                verb_role_impact = torch.autograd.Variable(verb_role_impact)

            role_predict = model(img, verb, verb_role_impact)

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
    parser.add_argument('--model', type=str, default='baseline_qctx_joint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--clip_norm', type=float, default=0.25)
    parser.add_argument('--num_workers', type=int, default=3)

    parser.add_argument('--baseline_model', type=str, default='', help='Pretrained baseline topdown model')
    parser.add_argument('--qctx_model', type=str, default='', help='Pretrained qctx topdown model')

    args = parser.parse_args()

    n_epoch = args.epochs
    batch_size = args.batch_size
    clip_norm = args.clip_norm
    n_worker = args.num_workers

    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    train_set = json.load(open(dataset_folder + '/' + args.train_file))

    encoder = imsitu_encoder.imsitu_encoder(train_set)

    train_set = imsitu_loader.imsitu_loader_impactfactor(imgset_folder, train_set, encoder,'train', encoder.train_transform)

    constructor = 'build_top_down_baseline'
    baseline = getattr(top_down_baseline, constructor)(encoder.get_num_roles(),encoder.get_num_verbs(), encoder.get_num_labels(), encoder)


    constructor = 'build_top_down_query_context_only_baseline'
    qctx_model = getattr(top_down_query_context_pretrained_baseline, constructor)(encoder.get_num_roles(),
                                                                                  encoder.get_num_verbs(), encoder.get_num_labels(), encoder, baseline)



    constructor = 'build_%s' % args.model
    model = getattr(baseline_qctx_weighted_joint_eval, constructor)(baseline, qctx_model)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder + '/' + args.dev_file))
    dev_set = imsitu_loader.imsitu_loader_impactfactor(imgset_folder, dev_set, encoder, 'val', encoder.dev_transform)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder + '/' + args.test_file))
    test_set = imsitu_loader.imsitu_loader_impactfactor(imgset_folder, test_set, encoder, 'test', encoder.dev_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(args.seed)
    if args.gpuid >= 0:
        model.cuda()
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True


    #load models
    utils.load_net(args.baseline_model, [model.baseline_model])
    utils.load_net(args.qctx_model, [model.qctx_model])



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
        role_dict = top1.all_res
        fail_val_all = top1.value_all_dict
        pass_val_dict = top1.vall_all_correct

        '''with open(args.model_saving_name+'_role_pred_data.json', 'w') as fp:
            json.dump(role_dict, fp, indent=4)'''

        '''with open(args.model_saving_name+'_fail_val_all.json', 'w') as fp:
            json.dump(fail_val_all, fp, indent=4)

        with open(args.model_saving_name+'_pass_val_all.json', 'w') as fp:
            json.dump(pass_val_dict, fp, indent=4)'''

        print('Writing predictions to file completed !')


    elif args.evaluate_rare:

        org_train_set = json.load(open(dataset_folder + '/' + args.org_train_file))
        #compute sparsity statistics
        verb_role_noun_freq = {}
        for image,frames in org_train_set.items():
            v = frames["verb"]
            items = set()
            for frame in frames["frames"]:
                for (r,n) in frame.items():
                    key = (v,r,n)
                    items.add(key)
            for key in items:
                if key not in verb_role_noun_freq: verb_role_noun_freq[key] = 0
                verb_role_noun_freq[key] += 1
                #per role it is the most frequent prediction
                #and among roles its the most rare

        org_eval_dataset = json.load(open(dataset_folder + '/' + args.org_test_file))
        image_sparsity = {}
        for image,frames in org_eval_dataset.items():
            v = frames["verb"]
            role_max = {}
            for frame in frames["frames"]:
                for (r,n) in frame.items():
                    key = (v,r,n)
                    if key not in verb_role_noun_freq: freq = 0
                    else: freq = verb_role_noun_freq[key]
                    if r not in role_max or role_max[r] < freq: role_max[r] = freq
            min_val = -1
            for (r,f) in role_max.items():
                if min_val == -1 or f < min_val: min_val = f
            image_sparsity[image] = min_val

        sparsity_max = 10
        x = range(0, sparsity_max+1)
        print ("evaluating images where most rare verb-role-noun in training is x , s.t. {} <= x <= {}".format(0, sparsity_max))
        n = 0
        for (k,v) in image_sparsity.items():
            if v in x:
                n+=1
        print ("total images = {}".format(n))

        top1, top5, val_loss = eval_rare(model, test_loader, encoder, args.gpuid, image_sparsity)

        top1_avg = top1.get_average_results(range(0, sparsity_max+1))
        top5_avg = top5.get_average_results(range(0, sparsity_max+1))

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
        avg_score /= 8

        print ('Test rare average :{:.2f} {} {}'.format( avg_score*100,
                                                         utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                         utils.format_dict(top5_avg, '{:.2f}', '5-')))




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









if __name__ == "__main__":
    main()












