import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, mode):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category, mode):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        dataset_name = model_path.split('/')[1]

        if 'new' in dataset_name:
            model_path = model_path.replace(dataset_name, 'ShapeNet')
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
        
        try:
            points_dict = np.load(file_path)
        except:
            import ipdb; ipdb.set_trace()
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        
        if 'df_value' in points_dict:
            distance_value = points_dict['df_value']
            distance_value = distance_value.astype(np.float32)
        else: # ShapeNet
            distance_path = os.path.join(model_path, 'df.npy')
            distance_value = np.load(distance_path).astype(np.float32)
        
        data = {
            None: points,
            'df': distance_value,
        }
        
        if self.transform is not None:
            data = self.transform(data)

        return data

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category, mode):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)
        
        try:
            points = pointcloud_dict['points'].astype(np.float32)
        except:
            points = pointcloud_dict['arr_0'].astype(np.float32)
        
        data = {
            None: points,
        }

        if self.transform is not None:
            data = self.transform(data)
        
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    ''' Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    '''
    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.5, partial_type='centery_random'):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio
        self.partial_type = partial_type

    def load(self, model_path, idx, category, mode):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if mode in ['val', 'test']: # fix the size in evaluation
            self.partial_type = 'centerz' if 'centerz' in self.partial_type else 'centery'
            self.part_ratio = 0.5

        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
        
        try:
            pointcloud_dict = np.load(file_path)
        except:
            print('Wrong file:', file_path)
            
        try:
            points = pointcloud_dict['points'].astype(np.float32)
        except:
            points = pointcloud_dict['arr_0'].astype(np.float32)
        
        
        if 'centery' in self.partial_type:

            if self.partial_type == 'centery_random':
                random_ratio = self.part_ratio * np.random.random()
            else:
                random_ratio = self.part_ratio

            # y is up-axis
            min_x = points[:,0].min()
            max_x = points[:,0].max()

            min_z = points[:,2].min()
            max_z = points[:,2].max()


            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_z = (max_z - min_z) * random_ratio

            center_x = (min_x + max_x) / 2
            center_z = (min_z + max_z) / 2
            start_x = center_x - (remove_size_x / 2)
            start_z = center_z - (remove_size_z / 2)

            crop_x_idx = np.where((points[:,0] < (start_x + remove_size_x)) & (points[:,0] > start_x))[0]
            crop_z_idx = np.where((points[:,2] < (start_z + remove_size_z)) & (points[:,2] > start_z))[0] 

            crop_idx = np.intersect1d(crop_x_idx, crop_z_idx)
            
            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0
       
            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }
      
        elif 'centerz' in self.partial_type:

            if self.partial_type == 'centerz_random':
                random_ratio = self.part_ratio * np.random.random()
            else:
                random_ratio = self.part_ratio

            # z is up-axis
            min_x = points[:,0].min()
            max_x = points[:,0].max()

            min_y = points[:,1].min()
            max_y = points[:,1].max()

            random_ratio = self.part_ratio * np.random.random()

            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_y = (max_y - min_y) * random_ratio

            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            start_x = center_x - (remove_size_x / 2)
            start_y = center_y - (remove_size_y / 2)


            crop_x_idx = np.where((points[:,0] < (start_x + remove_size_x)) & (points[:,0] > start_x))[0]
            crop_y_idx = np.where((points[:,1] < (start_y + remove_size_y)) & (points[:,1] > start_y))[0] 

            crop_idx = np.intersect1d(crop_x_idx, crop_y_idx)
            
            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0
       
            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }
       
        elif self.partial_type == 'randomy_random':
            # random location, random size
            min_x = points[:,0].min()
            max_x = points[:,0].max()

            min_z = points[:,2].min()
            max_z = points[:,2].max()
            
            random_ratio = self.part_ratio * np.random.random()

            remove_size_x = (max_x - min_x) * random_ratio
            remove_size_z = (max_z - min_z) * random_ratio

            start_x = min_x + (max_x - min_x - remove_size_x) * np.random.random()
            start_z = min_z + (max_z - min_z - remove_size_z) * np.random.random()

            crop_x_idx = np.where((points[:,0] < (start_x + remove_size_x)) & (points[:,0] > start_x))[0]
            crop_z_idx = np.where((points[:,2] < (start_z + remove_size_z)) & (points[:,2] > start_z))[0] 

            crop_idx = np.intersect1d(crop_x_idx, crop_z_idx)
            
            valid_mask = np.ones(len(points), dtype=bool)
            valid_mask[crop_idx] = 0
       
            remain_points = points[valid_mask]

            data = {
                None: remain_points,
            }
        
        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

