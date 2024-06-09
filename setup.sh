#!/bin/bash

# Create conda environment. Mamba is recommended for faster installation.
conda_env_name=seeing-unseen

mamba create -n $conda_env_name python=3.10 cmake=3.14.0 -y

# Install this repo as a package
mamba activate $conda_env_name
pip install -e .

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/openai/glide-text2im.git

cd seeing_unseen/third_party/

# Install Detic
cd Detic/
pip install -e .

# Install LLaVA
cd LLaVA/
pip install -e .

# Install SAM
cd Inpaint-Anything/segment_anything/
pip install -e .

# Install LAMA
cd Inpaint-Anything/lama/
pip install -e .

