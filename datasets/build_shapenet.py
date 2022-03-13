import os
import numpy as np
import glob
import open3d as o3d
from sklearn.neighbors import KDTree
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./data/ShapeNet')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--split', type=int, default=1)

args = parser.parse_args()



def write_ply(fn, point, normal=None, color=None):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)

    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)

    o3d.io.write_point_cloud(fn, ply)

    return

def generate_df_value(pc, points, occ):

    tree = KDTree(pc)
    nearest_dist, nearest_ind = tree.query(points, k=1)

    outside_ind = np.where(occ == 0)[0]
    nearest_dist[outside_ind] = -nearest_dist[outside_ind]

    return np.squeeze(nearest_dist)


object_list = glob.glob(args.dataest_path + '/*')
object_list.sort()

process_num = len(object_list) // args.split
print('current idx:', args.idx)
process_list = object_list[args.idx*process_num:(args.idx+1)*process_num]

for o in process_list:
    if 'yaml' in o:
        continue

    instance_list = glob.glob(o+'/*')
    instance_list.sort()

    for i in instance_list:
        if 'lst' in i:
            continue
        print(i)
        pointcloud_dict = np.load(os.path.join(i, 'pointcloud.npz'))

        pc = pointcloud_dict['points'].astype(np.float32)

        points_dict = np.load(os.path.join(i, 'points.npz'))
        points = points_dict['points']
        occupancies = np.unpackbits(points_dict['occupancies'])

        df_value = generate_df_value(pc, points, occupancies) 
        df_path = os.path.join(i, 'df.npy')

        np.save(df_path, df_value)

