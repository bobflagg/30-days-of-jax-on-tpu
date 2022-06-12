#!/usr/bin/env bash

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p $HOME/anaconda

export PATH=$HOME/anaconda/bin:$PATH
conda init

cd $HOME
source .bashrc

pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd
git clone https://github.com/google/flax.git
pip install --user -e flax

pip install torch
pip install torchvision
pip install tensorflow




