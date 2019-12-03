#!/bin/bash
module load python/3.7

module load cuda/10.1 cuda-10.1/cudnn/7.6

source $HOME/torchenv/bin/activate

python train.py > $SLURM_TMPDIR/log_CVAE01_MNIST.txt

cp $SLURM_TMPDIR/log_CVAE01_MNIST.txt .
cp $SLURM_TMPDIR/CVAE01_MLP_model.pt .
cp $SLURM_TMPDIR/*.png .