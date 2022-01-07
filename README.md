# Segmentation


Segmentation project to provide accurate clothing segmentation.

Datasets used - 
DeepFashion2 https://github.com/switchablenorms/DeepFashion2
Imaterialist https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data

## Quick Start

`pip install requirements.txt` to install all requirements needed for project.

Utilize `deepfashion2_to_coco.py` file to generate coco-type annotations. In case of current code, it stores it in `data/fs2` repo.

Utilize `experiments/dataset_generation.ipynb` to generate image and mask files from annotations to allow training of models.

`experiments/dataset_exploration_segmentation.ipynb` offers a look into the segmentation masks and images, and a summary of dataset stats.

`experiments/fashion_segmentation_training_smp.ipynb` has code for training on DeepFashion2 using smp. Utilize FPN net and se_resnext50_32x4d encoder with Imagenet weights for training. Results were mixed due to dataset being unpredictable at times. Clothing often cut off or occluded.

`experiments/dataset_exploration_segmentation_imaterialist.ipynb` has dataset exploration and dataloader code on imaterialist dataset

`pipeline` folder has training code on imaterialist dataset. Setup with config file to easily create a pipeline

Trained unet model based on Timm implementation of EfficientNet-B3 is available here: `https://sanats2.s3.us-east-2.amazonaws.com/unet-timm-efficient-b3.pth`
