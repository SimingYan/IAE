## Prepare Datasets

#### ShapeNet(114G)

Please download the dataset by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks, and put it under `data/ShapeNet` folder.

Since the original dataset only contains occupancy grid label, to add SDF label, please run
```
python build_shapenet.py --dataset_path=/path/to/ShapeNet
```

#### ScanNet(106G)

You can download our preprocessed dataset [here](https://drive.google.com/file/d/1iNN_uPp5wbzmBqtHRsJ0Z8KVzJV_1VIw/view?usp=sharing).


To build the dataset by yourself, please download ScanNet v2 dataset from the [official website](https://github.com/ScanNet/ScanNet). And run
```
python build_scannet.py --dataset_path=/path/to/scannet/scans --output_path=../data/scannet/rooms_01
```

Our data split is the same as [votenet](https://github.com/facebookresearch/votenet/tree/main/scannet/meta_data). To generate the datalist, please run
```
python build_scannet_datalist.py --votenet=/path/to/votenet/meta_data --dataset_path=../data/scannet/rooms_01  
```

