import torch
import numpy as np
import pdb

def cross_entropy_loss(pred, target, ignore_index=None):
    '''
    claculates cross entropy loss
    '''
    if ignore_index is not None and target == ignore_index:
        return 0
    _x_class = - pred[target]
    pred = torch.unsqueeze(pred,1)
    score,_ = torch.sort(pred, 0, True)
    max_score = torch.unsqueeze(score[0],1)

    max_score_broadcast = max_score.view(-1,1).expand(pred.size())
    denom = max_score + torch.log(torch.sum(torch.exp(pred - max_score_broadcast)))

    loss = _x_class + denom
    return loss

def load_net(fname, net_list, prefix_list = None):
    '''
    loading a pretrained model weights from a file
    '''
    need_modification = False
    if prefix_list is not None and len(prefix_list) > 0:
        need_modification = True
    for i in range(0, len(net_list)):
        if not torch.cuda.is_available():
            dict = torch.load(fname, map_location='cpu')
        else:
            dict = torch.load(fname)

        try:
            for k, v in net_list[i].state_dict().items():
                if need_modification:
                    k = prefix_list[i] + '.' + k

                if k in dict:
                    param = torch.from_numpy(np.asarray(dict[k].cpu()))
                    v.copy_(param)
                else:
                    print('[Missed]: {}'.format(k), v.size())
        except Exception as e:
            print(e)
            pdb.set_trace()
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

def set_trainable(model, requires_grad):
    '''
    set model parameters' training mode on/off
    '''
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad

def format_dict(d, s, p):
    '''
    format the performance metrics according to original ImSitu format
    '''
    rv = ""
    for (k,v) in d.items():
        if len(rv) > 0: rv += " , "
        rv+=p+str(k) + ":" + s.format(v*100)
    return rv