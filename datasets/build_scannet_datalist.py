import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--votenet', type=str, default='./votenet/scannet/meta_data/')
parser.add_argument('--dataset_path', type=str, default='../../data/scannet/rooms_01/')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--split', type=int, default=1)

args = parser.parse_args()

data_list = glob.glob(args.dataset_path + '*')

data_list.sort()

train_file = open(args.dataset_path + 'train.lst', 'w')
test_file = open(args.dataset_path + 'test.lst', 'w')
val_file = open(args.dataset_path + 'val.lst', 'w')

votenet_train = open(args.votenet + '/scannetv2_train.txt', 'r').readlines()
votenet_test = open(args.votenet + '/scannetv2_test.txt', 'r').readlines()
votenet_val = open(args.votenet + '/scannetv2_val.txt', 'r').readlines()

votenet_train_list = []
votenet_test_list = []
votenet_val_list = []

for i in range(len(votenet_train)):
    votenet_train_list.append(votenet_train[i].strip())

for i in range(len(votenet_test)):
    votenet_test_list.append(votenet_test[i].strip())

for i in range(len(votenet_val)):
    votenet_val_list.append(votenet_val[i].strip())

for i in range(len(data_list)):
    scene_name = data_list[i].split('/')[-1]
    main_scene_name = '_'.join(scene_name.split('_')[:-1])
    if main_scene_name in votenet_train_list:
        train_file.write(scene_name + '\n')
    elif main_scene_name in votenet_test_list:
        test_file.write(scene_name + '\n')  
    elif main_scene_name in votenet_val_list:
        val_file.write(scene_name + '\n')
    

train_file.close()
test_file.close()
val_file.close()
