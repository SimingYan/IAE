#  Ref: https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/utils/Dataset_Loc.py

def Dataset_Loc(dataset):
    def fetch_files(filelist):
        return [item.strip() for item in open(filelist).readlines()]

    dataset = dataset.lower()

    if dataset == 'modelnet40':
        '''Actually we find that using data from PointNet++: '''
        NUM_CLASSES = 40
        VALID_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/test_files.txt')
        TRAIN_FILES = fetch_files('./data/modelnet40_ply_hdf5_2048/train_files.txt')
    else:
        raise ValueError('dataset not exists')

    return NUM_CLASSES, TRAIN_FILES, VALID_FILES
