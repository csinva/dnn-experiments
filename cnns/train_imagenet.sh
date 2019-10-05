#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:1
module load python
# module load pytorch
# python3 train_imagenet.py -a alexnet --save-dir /scratch/users/vision/chandan/cnns/alexnet --lr 0.01 /scratch/users/vision/data/cv/imagenet_full/ # alexnet needs a lower lr
python3 train_imagenet.py -a resnet18 --batch-size 16 --save-dir /scratch/users/vision/chandan/cnns/resnet18 /scratch/users/vision/data/cv/imagenet_full/
