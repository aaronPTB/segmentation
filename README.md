# Segmentation


Segmentation project to provide accurate clothing segmentation.

Dataset used - DeepFashion2 https://github.com/switchablenorms/DeepFashion2

## Quick Start

`pip install requirements.txt` to install all requirements needed for project.

Utilize `deepfashion2_to_coco.py` file to generate coco-type annotations. In case of current code, it stores it in `data/fs2` repo.

Utilize `experiments/dataset_generation.ipynb` to generate image and mask files from annotations to allow training of models.

`experiments/dataset_exploration_segmentation.ipynb` offers a look into the segmentation masks and images, and a summary of dataset stats.

`experiments/fashion_segmentation_training_smp.ipynb` has code for training using smp. Utilize FPN net and se_resnext50_32x4d encoder with Imagenet weights for training. 
