#  Ref: https://github.com/hansen7/OcCo/tree/master/OcCo_Torch/utils

import numpy as np

"""
	================================================
	=== Library for Point Cloud Utility Function ===
	================================================
"""


def pc_normalize(pc):
    """ Normalise the Input Point Cloud into a Unit Sphere """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """ A Simple Yet Inefficient Farthest Point Sampling on Point Cloud """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def random_shift_point_cloud(batch_data, shift_range=0.1):
    """ Shift the Point Cloud along the XYZ axis, magnitude is randomly sampled from [-0.1, 0.1] """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Scale the Point Cloud Objects into a Random Magnitude between [0.8, 1.25] """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """ Randomly Dropout out a Portion of Points, Ratio is Randomly Selected between [0, 0.875]	"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set the rest as the first point
    return batch_pc


