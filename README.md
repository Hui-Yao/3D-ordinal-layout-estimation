# environment configuration
## create conda env
conda create -n ordinal python=3.6
## activate conda env
conda activate ordinal
## install pytorch
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
## install dependencies
conda env create -f env.yaml





# train


# evaluate

# predict
python predict.py --image_path path_to_image --pretrained_path path_to_predtrained_model



