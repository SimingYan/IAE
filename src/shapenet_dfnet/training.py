import os
import torch
from src.common import compute_iou, add_key
from src.training import BaseTrainer

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
    '''

    def __init__(self, model, optimizer, device=None, vis_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.vis_dir = vis_dir

        self.loss = torch.nn.L1Loss(reduction='none')

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        eval_dict = {}
        
        points = data.get('points').to(device)
        df = data.get('points.df').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)

        points_iou = data.get('points_iou').to(device)
        df_iou = data.get('points_iou.df').to(device)
        
        batch_size = points.size(0)

        kwargs = {}
        
        # add pre-computed index
        inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get('points.normalized'), 'p', 'p_n', device=device)
        points_iou = add_key(points_iou, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            p_out = self.model(points_iou, inputs, **kwargs)
        
        df_iou_np = (df_iou >= -0.1).cpu().numpy()
        df_iou_hat_np = (p_out >= -0.1).cpu().numpy()
        
        iou = compute_iou(df_iou_np, df_iou_hat_np).mean()
        
        eval_dict['iou'] = iou

        return eval_dict

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        df = data.get('points.df').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        
        c = self.model.encode_inputs(inputs)

        kwargs = {}
        # General points
        output = self.model.decode(p, c, **kwargs) 
        
        loss = self.loss(output, df).sum(-1).mean()

        return loss

