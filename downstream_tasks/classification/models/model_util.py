import pdb, torch, torch.nn as nn, torch.nn.functional as F
from models.dgcnn_cls import DGCNN_Cls_Encoder

class encoder(nn.Module):
    def __init__(self, num_channel=3, out_dim=1024, backbone='dgcnn_cls'):
        super(encoder, self).__init__()

        if backbone == 'dgcnn_cls':
            self.encoder = DGCNN_Cls_Encoder(c_dim=out_dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        feat = self.encoder(x)
        if len(feat.shape) == 3:
            feat = feat.squeeze()
        return feat

