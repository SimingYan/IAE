from src.encoder import (
    pointnet2_det, dgcnn_semseg, dgcnn_cls
)


encoder_dict = {
    'pointnet2': pointnet2_det.Pointnet2,
    'dgcnn_semseg': dgcnn_semseg.DGCNN_semseg,
    'dgcnn_cls': dgcnn_cls.DGCNN_cls,
}
