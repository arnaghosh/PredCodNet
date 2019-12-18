#!/bin/bash
module load python/3.7

module load cuda/10.1 cuda-10.1/cudnn/7.6

source $HOME/torchenv/bin/activate

python train_limited_labels_MNIST.py > $SLURM_TMPDIR/log_CVAE_Ztfo2++_MNIST_acc.txt

cp $SLURM_TMPDIR/log_CVAE_Ztfo2++_MNIST_acc.txt .