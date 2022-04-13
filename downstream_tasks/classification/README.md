## Dataset

We evaluate shape classification task on ModelNet40 dataset. Please download [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip it under `data` folder.

## Pretrained Models
We provide our pretrained models and training log here: [Google Drive](https://drive.google.com/drive/folders/1eiD99VQ04OHMwVnefrgkKS3YcMg2DTox?usp=sharing).

Please download and put them under `pretrained_models` folder.

### Linear evaluation on ModelNet40

```
python train_svm.py --encoder=dgcnn_cls --restore_path=./pretrained_models/modelnet40_svm.pt
```

### Fine-tuning on ModelNet40
```
python train_cls.py --use_sgd --model=dgcnn_clsft --dataset=modelnet40 --log_dir=ours_clsft --restore --restore_path=./pretrained_models/modelnet40_clsft.pt
```

## Acknowledgements

We would like to thank and acknowledge referenced codes from

OcCo: https://github.com/hansen7/OcCo.
