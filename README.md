# Simplest Deep Learning Regression Framework

This repository provides simplest framework for designing regression neural networks
It serves as a starting point for more complex projects, where more models and loss functions could be added

If you are interested in using such framework to address regression problems then check out: [http://github.com/masadcv/SimplestDeepLearningClassifier.git](http://github.com/masadcv/SimplestDeepLearningClassifier.git)

## Setup
To setup, simply clone this repository and install dependencies:

```
$ git clone http://github.com/masadcv/SimplestDeepLearningRegressor.git
$ cd SimplestDeepLearningRegressor
$ pip3 install -r requirements.txt
```

## Training
To train a network, provide a config following format provided in `./configs` folder or use an existing config with training script:

```
$ python3 training.py --model mnistmlp --config configs/config_l1.json
```

## Evaluation
To evaluate a pretrained network, provide the folder containing training output with evaluation script:

```
$ python3 evaluation.py --folder /path/to/training/folder
```

The evaluation output will be saved in `/path/to/training/folder`

## Training Monitoring
This framework provides tensorboard interface for monitoring training. To check training status, running tensorboard using:

```
$ tensorboard --logdir /path/to/training/folder --port 6006
```

Following this, tensorboard session will be accessible in browser at: [localhost:6006](localhost:6006)

## Batched Training/Evaluation Scripts
To train all supported models configurations, run batched training script as:
```
$ source scripts/train_all.sh
```

To evaluated all trained models run batched evaluation script as:
```
$ source scripts/eval_all.sh
```
