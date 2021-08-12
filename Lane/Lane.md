# Lane Segmentation

![Teaser](./Teaser.gif)

## 1. Create environment
  + Create conda environment with following command `conda create -y -n lane python=3.8`
  + Activate environment with following command `conda activate lane`
  + Install requirements with following command `pip install -r requirements.txt`

## 2. Inference
  + Prepare your image folder
  + `python run/Inference.py --source [IMAGE_FOLDER_DIR]`

## Performance - KAIST Highway Dataset
  + Maximum F1 Score: 94.8
  + Maximum IoU: 88.5
