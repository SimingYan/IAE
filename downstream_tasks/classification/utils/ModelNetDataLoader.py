import os, sys, torch, h5py, warnings, numpy as np
import open3d as o3d

from torch.utils.data import Dataset
sys.path.append('utils')
warnings.filterwarnings('ignore')

class General_CLSDataLoader_HDF5(Dataset):
    def __init__(self, file_list, num_point=1024, data_aug=0):
        self.num_point = num_point
        self.file_list = file_list
        self.points_list = np.zeros((1, num_point, 3))
        self.labels_list = np.zeros((1,))
        self.data_aug = data_aug

        for file in self.file_list:
            data, label = self.loadh5DataFile(file)
            self.points_list = np.concatenate(
                [self.points_list, data[:, :self.num_point, :]], axis=0)
            self.labels_list = np.concatenate([self.labels_list, label.ravel()], axis=0)
        
        self.points_list = self.points_list[1:]
        self.labels_list = self.labels_list[1:]
        assert len(self.points_list) == len(self.labels_list)
        print('Number of Objects: ', len(self.labels_list))

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __len__(self):
        return len(self.points_list)

    def __getitem__(self, index):
        point_xyz = self.points_list[index][:, 0:3]

        point_label = self.labels_list[index].astype(np.int32)

        return point_xyz, point_label


