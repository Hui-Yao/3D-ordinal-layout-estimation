# Introduction
3D layout estimation of general rooms based on ordinal semantic segmentation.



# Installation
The code is tested with Ubuntu 18.04, PyTorch v1.6, CUDA 10.1 and cuDNN v7.6.

```
## create conda env
conda create -n ordinal python=3.6
## activate conda env
conda activate ordinal
## install pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
## install dependencies
conda env create -f env.yaml
```
# data preparation
You can download the InteriorNet-layout dataset here: 


# training
Run the following command to train our network:
```
 python main.py --data_path path-to-the-dataset --model_name the-name-of-a-new-training
```
# pre-trained model
You can download our pre-trained models here: 

# evaluation
Run the following command to evaluate the performance:
```
python evaluate.py --data_path path_to_testing_set --pretrained_path path_to_predtrained_model
```

# prediction
Run the following command to predict on a single image:
```
python predict.py --image_path path_to_image --pretrained_path path_to_predtrained_model
```


