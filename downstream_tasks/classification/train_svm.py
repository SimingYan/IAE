#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://scikit-learn.org/stable/modules/svm.html
#  Ref: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
# CUDA_VISIBLE_DEVICES=2 python train_svm.py --model=vrcnet_util --encoder=dgcnn_cls --output_dim=1024 --restore_path=/path/to/ckpt

import os, sys, torch, argparse, datetime, importlib, numpy as np
sys.path.append('utils')
sys.path.append('models')
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ModelNetDataLoader import General_CLSDataLoader_HDF5
from Torch_Utility import copy_parameters
# from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from Dataset_Loc import Dataset_Loc
from sklearn import svm, metrics
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('SVM on Point Cloud Classification')

    ''' === Network Model === '''
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--model', default='pcn_util', help='model [default: pcn_util]')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--restore_path', type=str, default='', help="path to pre-trained weights [default: None]")
    parser.add_argument('--grid_search', action='store_true', help='opt parameters via Grid Search [default: False]')
    parser.add_argument('--encoder', type=str, default='dgcnn_cls', help="which encoder I will use: [PCN, DGCNN]")
    parser.add_argument('--output_dim', type=int, default=1024, help="output feature dim")
    parser.add_argument('--num_points', type=int, default=2048, help="output feature dim")

    ''' === Dataset === '''
    parser.add_argument('--partial', action='store_true', help='partial objects [default: False]')
    parser.add_argument('--bn', action='store_true', help='with background noise [default: False]')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset [default: modelnet40]')
    parser.add_argument('--fname', type=str, default="", help='filename, used in ScanObjectNN [default: ]')
    parser.add_argument('--switch_yz', action='store_true', help='for conv_onet model which takes y axis as height')

    return parser.parse_args()


def train_svm(trainDataLoader, testDataLoader, encoder, model_dict=None): 

    if model_dict:
        encoder = copy_parameters(encoder, model_dict, verbose=True)

    X_train, y_train, X_test, y_test = [], [], [], []
    with torch.no_grad():
        encoder.eval()
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            if points.shape[2] == 6:
                points = points[:, :, :3]

            points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
            feats = encoder(points)
            X_train.append(feats.cpu().numpy())
            y_train.append(target.cpu().numpy())

        for points, target in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
            if points.shape[2] == 6:
                points = points[:, :, :3]

            points, target = points.float().transpose(2, 1).cuda(), target.long().cuda()
            feats = encoder(points)
            X_test.append(feats.cpu().numpy())
            y_test.append(target.cpu().numpy())


    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

    # Optional: Standardize the Feature Space
    # X_train, X_test = scale(X_train), scale(X_test)
    #''' === Simple Trial === '''
    #linear_svm = svm.SVC(kernel='linear')
    #linear_svm.fit(X_train, y_train)
    #y_pred = linear_svm.predict(X_test)
    #print("Simple Linear SVm accuracy:", metrics.accuracy_score(y_test, y_pred), "\n") 
    #accuracy_1 = metrics.accuracy_score(y_test, y_pred)

    clf = svm.LinearSVC(random_state=0)
    clf.fit(X_train, y_train)  
    result = clf.predict(X_test)  
    accuracy = np.sum(result==y_test).astype(float) / np.size(y_test)
    print("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))

    #max_accuracy = max(accuracy_1, accuracy_2)
    #return max_accuracy
    return accuracy

if __name__ == '__main__':
    args = parse_args()

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_dict = checkpoint['state_dict']
        else:
            model_dict = checkpoint
    else:
        model_dict = None
    
    
    if 'STRL' in args.restore_path:
        model_dict = {k.replace('online_network', 'encoder'): v for k, v in model_dict.items()} 
        
    if args.encoder == 'PCN_Onet':
        model_dict = {k.replace('encoder.pcn_encoder', 'encoder'): v for k, v in model_dict.items()} 
    if args.encoder == 'PointnetPlusPlus_v2':
        model_dict = {k.replace('encoder.pointnet2_encoder', 'encoder'): v for k, v in model_dict.items()} 

    if args.dataset == 'modelnet40':
        _, TRAIN_FILES, TEST_FILES = Dataset_Loc(dataset='modelnet40', fname='',
                                             partial=False, bn=False)
        TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES, num_point=args.num_points, switch_yz=False)
        TEST_DATASET = General_CLSDataLoader_HDF5(file_list=TEST_FILES, num_point=args.num_points, switch_yz=False)
        trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=8, shuffle=False, num_workers=4)
        testDataLoader = DataLoader(TEST_DATASET, batch_size=8, shuffle=False, num_workers=4)
    elif args.dataset == 'modelnet40_normal':
        from torchvision import transforms
        import data_utils as d_utils
        from ModelNet40Loader import ModelNet40Cls

        transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
                d_utils.PointcloudScale(),
                d_utils.PointcloudTranslate(),
                d_utils.PointcloudJitter(),
            ]
        )
        #trainset = ModelNet40Cls(2048, train=True, transforms=transforms, download=False)
        trainset = ModelNet40Cls(2048, train=True, transforms=None, download=False)
        testset = ModelNet40Cls(2048, train=False, transforms=None, download=False)
        
        print(len(trainset))
        print(len(testset))
        
        if 1:
            trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
            testDataLoader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        else:
            trainDataLoader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=False, num_workers=8)
            testDataLoader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False, num_workers=8)
 
    MODEL = importlib.import_module('vrcnet_util')
    encoder = MODEL.encoder(num_channel=3, out_dim=args.output_dim, backbone=args.encoder).to(device)
    
    nparameters = sum(p.numel() for p in encoder.parameters())
    print('Total number of parameters: %d' % nparameters)

   
    svm_acc = train_svm(trainDataLoader, testDataLoader, encoder, model_dict)

    print('max acc:', svm_acc)

