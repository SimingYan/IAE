#  Ref: https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/train_svm.py
#  Ref: https://scikit-learn.org/stable/modules/svm.html
#  Ref: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

import os, sys, torch, argparse, datetime, importlib, numpy as np
sys.path.append('utils')
sys.path.append('models')
from ModelNetDataLoader import General_CLSDataLoader_HDF5
from Torch_Utility import copy_parameters
from torch.utils.data import DataLoader
from Dataset_Loc import Dataset_Loc
from sklearn import svm, metrics
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser('SVM on Point Cloud Classification')

    ''' === Network Model === '''
    parser.add_argument('--model', default='pcn_util', help='model [default: pcn_util]')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size [default: 24]')
    parser.add_argument('--restore_path', type=str, default='', help="path to pre-trained weights [default: None]")
    parser.add_argument('--encoder', type=str, default='dgcnn_cls', help="which encoder I will use: [PCN, DGCNN]")
    parser.add_argument('--output_dim', type=int, default=1024, help="output feature dim")
    parser.add_argument('--num_points', type=int, default=2048, help="output feature dim")

    ''' === Dataset === '''
    parser.add_argument('--dataset', type=str, default='modelnet40', help='dataset [default: modelnet40]')

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

    clf = svm.LinearSVC(random_state=0)
    clf.fit(X_train, y_train)  
    result = clf.predict(X_test)  
    accuracy = np.sum(result==y_test).astype(float) / np.size(y_test)
    print("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))

    return accuracy

if __name__ == '__main__':
    args = parse_args()
    
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
    
    if args.dataset == 'modelnet40':
        _, TRAIN_FILES, TEST_FILES = Dataset_Loc(dataset='modelnet40')
        TRAIN_DATASET = General_CLSDataLoader_HDF5(file_list=TRAIN_FILES, num_point=args.num_points)
        TEST_DATASET = General_CLSDataLoader_HDF5(file_list=TEST_FILES, num_point=args.num_points)
        trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
        testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    MODEL = importlib.import_module('model_util')
    encoder = MODEL.encoder(num_channel=3, out_dim=args.output_dim, backbone=args.encoder).to(device)
 
    nparameters = sum(p.numel() for p in encoder.parameters())
    print('Total number of parameters: %d' % nparameters)
   
    svm_acc = train_svm(trainDataLoader, testDataLoader, encoder, model_dict)

    print('max acc:', svm_acc)
