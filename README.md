# 3D-layout-estimation-of-general-rooms-based-on-ordinal-semantic-segmentation


# Installation
The code is tested with Ubuntu 18.04, PyTorch v1.6, CUDA 10.1 and cuDNN v7.6.
'''
## create conda env
conda create -n ordinal python=3.6
## activate conda env
conda activate ordinal
## install pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
## install dependencies
conda env create -f env.yaml
'''




# train
 python main.py --data_path path-to-the-dataset --model_name the-name-of-a-new-training

# evaluate
python evaluate.py --data_path path_to_testing_set --pretrained_path path_to_predtrained_model

# predict
python predict.py --image_path path_to_image --pretrained_path path_to_predtrained_model



