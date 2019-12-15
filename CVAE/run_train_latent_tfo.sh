#!/bin/bash
module load python/3.7

module load cuda/10.1 cuda-10.1/cudnn/7.6

source $HOME/torchenv/bin/activate

python train_latent_tfo_modif.py > $SLURM_TMPDIR/log_CVAE_Ztfo2++_MNIST.txt

cp $SLURM_TMPDIR/log_CVAE_Ztfo2++_MNIST.txt .
cp $SLURM_TMPDIR/CVAE_Ztfo2_MLP_model++.pt .
cp $SLURM_TMPDIR/*.png .