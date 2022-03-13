#  Ref: https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
#  Ref: https://github.com/hansen7/OcCo/tree/master/OcCo_Torch/utils


import torch, os, random, numpy as np

def copy_parameters(model, pretrained_dict, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    new_state_dict = {}
    for param_name in pretrained_dict:
        if 'encoder.dgcnn_encoder' in param_name:
            newname = param_name.replace('encoder.dgcnn_encoder', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        elif 'encoder.foldnet_encoder' in param_name:
            newname = param_name.replace('encoder.foldnet_encoder', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        elif 'module' in param_name:
            newname = param_name.replace('module', 'encoder')
            new_state_dict[newname] = pretrained_dict[param_name]
        else:
            new_state_dict[param_name] = pretrained_dict[param_name]

    pretrained_dict = new_state_dict
   
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def copy_parameters_ft(model, pretrained_dict, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
    new_state_dict = {}
    for param_name in pretrained_dict:
        if 'encoder.dgcnn_encoder' in param_name:
            newname = param_name.replace('encoder.dgcnn_encoder.', '')
            new_state_dict[newname] = pretrained_dict[param_name]
        else:
            new_state_dict[param_name] = pretrained_dict[param_name]

    pretrained_dict = new_state_dict
   
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def weights_init(m):
    """
    Xavier normal initialisation for weights and zero bias,
    find especially useful for completion and segmentation Tasks
    """
    classname = m.__class__.__name__
    if (classname.find('Conv1d') != -1) or (classname.find('Conv2d') != -1) or (classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum
