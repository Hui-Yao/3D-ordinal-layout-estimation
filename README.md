# Introduction
3D layout estimation of general rooms based on ordinal semantic segmentation.
## Overall architecture
![流程图](https://user-images.githubusercontent.com/52377012/162351737-4cc149ce-aa7e-4f57-92fd-c9cb9aded329.PNG)
## 3D reconstruction of room layout
![3333](https://user-images.githubusercontent.com/52377012/162351848-9dead60c-7b03-4f7a-bd29-67a880c574d9.PNG)


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
pip install -r requirements.txt
```
# Data preparation
You can download the InteriorNet-layout dataset here: https://github.com/Hui-Yao/InteriorNet-Layout/blob/main/README.md


# Training
Run the following command to train our network:
```
 python main.py --data_path path-to-the-dataset --model_name the-name-of-a-new-training
```
# Pre-trained model
You can download our pre-trained models here: https://drive.google.com/drive/folders/1bR4FFUFm7_eUEyav2fu8PUPlJP8i-gf3

# Evaluation
Run the following command to evaluate the performance:
```
python evaluate.py --data_path path_to_testing_set --pretrained_path path_to_predtrained_model
```

# Prediction
Run the following command to predict on a single image:
```
python predict.py --image_path path_to_image --pretrained_path path_to_predtrained_model
```


