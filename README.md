# FalconAIChallenge

The conda environment can be recreated using the following command.

`conda create -f environment.yaml`

To use the conda environment in a Jupyter notebook, run:

`python -m ipykernel install --user --name pytorch_env --display-name "Python (pytorch_env)"`


## Computer Vision Approach

Using OpenCV's template matching algorithm and custom cropping functions, champions in the teams can be detected.

Run the following command to use the computer vision approach:

`python match_template.py --config ./config.json`

## Deep Learning Approach

The last layer of a pretrained Resnet-18 model has been replaced and trained for an Object Detection task of predicting champions.

Run the following command to use the deep learning approach:

`python train.py --config ./config.json`
