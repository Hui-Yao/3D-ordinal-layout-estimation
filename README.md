# create conda env
conda create -n layout python=3.6
# activate conda env
conda activate layout
# install pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
# install dependencies
conda env create -f env.yaml
