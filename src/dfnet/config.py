import torch
import torch.distributions as dist
from torch import nn
import os
from src.encoder import encoder_dict
from src.dfnet import models, training
from src import data
from src import config
import numpy as np


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' 
    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']

    decoder = models.decoder_dict[decoder](
        c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )
    
    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = models.ConvolutionalDFNetwork(
        decoder, encoder, device=device
    )
   
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module)
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.Trainer(
        model, optimizer,
        device=device, 
        vis_dir=vis_dir, threshold=threshold,
    )

    return trainer

def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    
    input_type = cfg['data']['input_type']
    fields = {}
    if cfg['data']['points_file'] is not None:
        fields['points'] = data.PointsField(
            cfg['data']['points_file'], points_transform,
            unpackbits=False,
            multi_files=cfg['data']['multi_files'],
        )

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                unpackbits=False,
                multi_files=cfg['data']['multi_files'],
            )

    return fields
