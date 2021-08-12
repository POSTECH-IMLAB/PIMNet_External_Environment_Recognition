# Lane Segmentation

![Teaser](./Teaser.gif)

## 1. Create environment
  + Create conda environment with following command `conda create -y -n lane python=3.8`
  + Activate environment with following command `conda activate lane`
  + Install requirements with following command `pip install -r requirements.txt`

## 2. Preparation
  + Download checkpoint from [Link](https://drive.google.com/file/d/1DONSeQ43PwAnW-Eehpvo5UaRAJP4mhZy/view?usp=sharing)
  + Move file as follows `./snapshots/HighwayLane/latest.pth`. Create folder if needed.

## 3. Inference
  + Prepare your image folder
  + `python run/Inference.py --source [IMAGE_FOLDER_DIR]`

## Performance - KAIST Highway Dataset
  + Maximum F1 Score: 94.8
  + Maximum IoU: 88.5
  + Throughput: 43 fps
  + GPU Mem Usage: 1.5 GB
