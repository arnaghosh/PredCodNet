#!/bin/bash
module load python/3.7

module load cuda/10.1 cuda-10.1/cudnn/7.6

source $HOME/torchenv/bin/activate

python MNIST_train.py > $SLURM_TMPDIR/log_MNIST_CNN_10frac_1.txt

cp $SLURM_TMPDIR/log_MNIST_CNN_10frac_1.txt .
