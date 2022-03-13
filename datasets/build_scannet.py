import os
import time
import numpy as np
import argparse
import trimesh
from os.path import join
from os import listdir
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path', type=str, default='./data/scannet_raw/scans')
parser.add_argument('--output_path', type=str, default='./data/scannet')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--split', type=int, default=1)

args = parser.parse_args()

path_in = args.dataset_path
path_out = args.output_path

n_pointcloud_files = 5
points_dtype = np.float16
n_iou_points = 100000

if not os.path.exists(path_out):
    os.makedirs(path_out)

def write_ply(fn, point, normal=None, color=None):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(point)

    if color is not None:
        ply.colors = o3d.utility.Vector3dVector(color)

    if normal is not None:
        ply.normals = o3d.utility.Vector3dVector(normal)

    o3d.io.write_point_cloud(fn, ply)

    return

def align_axis(file_name, mesh):
    lines = open(file_name).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))

    mesh.apply_transform(axis_align_matrix)
    return mesh, axis_align_matrix

def sample_points(mesh, n_points=100000, p_type=np.float16):
    pcl, idx = mesh.sample(n_points, return_index=True)
    normals = mesh.face_normals[idx]

    out_dict = {
        'points': pcl.astype(p_type),
        'normals': normals.astype(p_type),
    }
    return out_dict

def scale_to_unit_cube(mesh, z_level=-0.5):
    bbox = mesh.bounds
    loc = (bbox[0] + bbox[1]) / 2
    
    scale = 1. / (bbox[1] - bbox[0]).max()
    vertices_t = (mesh.vertices - loc.reshape(-1, 3)) * scale
    z_min = min(vertices_t[:, 2])

    # create_transform_matrix
    S_loc = np.eye(4)
    S_loc[:-1, -1] = -loc
    # create scale mat
    S_scale = np.eye(4) * scale
    S_scale[-1, -1] = 1
    # create last translate matrix
    S_loc2 = np.eye(4)
    S_loc2[2, -1] = -z_min + z_level

    S = S_loc2 @ S_scale @ S_loc
    mesh.apply_transform(S)
    
    return mesh, S

def sample_iou_points(mesh, iou_file, padding=0.1, illustrate=False):
    points = (np.random.rand(n_iou_points, 3).astype(np.float32) - 0.5) * (1 + padding)

    mesh_vertices = mesh.vertices

    tree = KDTree(mesh_vertices)
    nearest_dist, nearest_ind = tree.query(points, k=1)
    
    df = np.squeeze(nearest_dist)
    occ = np.squeeze(nearest_dist < 0.005)

    points = points.astype(points_dtype)

    out_dict = {
        'points': points,
        'df_value': df,
        'occupancies': np.packbits(occ),
    }
    np.savez(iou_file, **out_dict)

    if illustrate:
        v = out_dict['points'][np.unpackbits(out_dict['occupancies']) == 1]
        write_ply(iou_file.replace('npz', 'ply'), v)
        write_ply(iou_file.replace('points_iou_', 'iou_').replace('npz', 'ply'), points)

def slice_mesh(mesh):
    bbox = mesh.bounds
    
    max_x = max(bbox[0][0], bbox[1][0])
    min_x = min(bbox[0][0], bbox[1][0])
    max_y = max(bbox[0][1], bbox[1][1])
    min_y = min(bbox[0][1], bbox[1][1])

    pointer_x = min_x
    
    mesh_list = []

    while(pointer_x + 3 < max_x):
        pointer_y = min_y   
        while(pointer_y + 3 < max_y):
            # x-axis slice
            s_point = np.asarray([pointer_x, 0, 0])
            s_normal = np.asarray([np.abs(pointer_x), 0, 0])
            segment_mesh = trimesh.intersections.slice_mesh_plane(mesh, s_normal, s_point)
            e_point = np.asarray([pointer_x + 3, 0, 0])
            e_normal = np.asarray([-np.abs(pointer_x + 3), 0, 0])
            segment_mesh = trimesh.intersections.slice_mesh_plane(segment_mesh, e_normal, e_point)

            # y-axis slice
            s_point = np.asarray([0, pointer_y, 0])
            s_normal = np.asarray([0, np.abs(pointer_y), 0])
            segment_mesh = trimesh.intersections.slice_mesh_plane(segment_mesh, s_normal, s_point)
            e_point = np.asarray([0, pointer_y + 3, 0])
            e_normal = np.asarray([0, -np.abs(pointer_y + 3), 0])
            segment_mesh = trimesh.intersections.slice_mesh_plane(segment_mesh, e_normal, e_point)

            mesh_list.append(segment_mesh)

            pointer_y = pointer_y + 0.5

        pointer_x = pointer_x + 0.5

    return mesh_list

def process(scene_name):

    print(scene_name)
    if os.path.exists(os.path.join(path_out, scene_name + '_00')):
        return

    # load mesh
    mesh = trimesh.load(os.path.join(path_in, scene_name, scene_name+'_vh_clean.ply'), process=False)
    txt_file = os.path.join(path_in, scene_name, '%s.txt' % scene_name)

    mesh, align_mat = align_axis(txt_file, mesh)

    mesh_list = slice_mesh(mesh)
    print('mesh number:', len(mesh_list))
    
    scale_mesh_list = []
    cnt = 0
    for m in mesh_list:
        if m.vertices.shape[0] < 70000:
            continue
        scale_m, scale_mat = scale_to_unit_cube(m)
        scale_mesh_list.append(scale_m)
        
        sub_scene_path = os.path.join(path_out, scene_name + '_%02d' % cnt)
        if not os.path.exists(sub_scene_path):
            os.mkdir(sub_scene_path)
        if not os.path.exists(os.path.join(sub_scene_path, 'pointcloud')):
            os.mkdir(os.path.join(sub_scene_path, 'pointcloud'))
        if not os.path.exists(os.path.join(sub_scene_path, 'points_iou')):
            os.mkdir(os.path.join(sub_scene_path, 'points_iou'))
        
        for i in range(n_pointcloud_files):
            pcl = sample_points(scale_m)
            out_file = os.path.join(sub_scene_path, 'pointcloud', 'pointcloud_%02d.npz' % i)

            iou_file = os.path.join(sub_scene_path, 'points_iou', 'points_iou_%02d.npz' % i)
            sample_iou_points(scale_m, iou_file, illustrate=False) 
            np.savez(out_file, **pcl)
        
        cnt += 1
    

file_list = listdir(path_in)
file_list.sort()

process_num = len(file_list) // args.split
process_list = file_list[args.idx*process_num:(args.idx+1)*process_num]

for f in process_list:
    process(f)
