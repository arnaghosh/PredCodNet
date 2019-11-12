#!/bin/bash
module load python/3.7

module load cuda/10.1 cuda-10.1/cudnn/7.6

source $HOME/torchenv/bin/activate

python train.py > $SLURM_TMPDIR/log_VAE_MNIST_3.txt

cp $SLURM_TMPDIR/log_VAE_MNIST_3.txt .
cp $SLURM_TMPDIR/VAE_MLP_model.pt .
cp $SLURM_TMPDIR/*.png .
