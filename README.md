# Finding Your (3D) Center: 3D object detection using a learned loss

Repository for the paper [Finding Your (3D) Center: 3D object detection using a learned loss](https://arxiv.org/abs/2004.02693). We provide an implementation for the single class approach reported in the paper for simplicity and clarity. We choose to do this as it offers better focus on our primary contribution, a new approach to model training. A multi-class pipeline will be provided in a separate repository which will also contain multiple improvements. In general this repository can be adapted with the addition of only a few extra files which can be supplied if required.

![Network overview](teaser.png)

## Requirements

The repository is built and tested with the following setup:

```
python == 3.6.8
cuda == 10.0
```

In addition there are a number python dependencies which can be installed through the `requirements.txt`. The most straight forward way to install these is through a virutal environment:

```
virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Custom Ops

The PointNet++ backbone requires the compilation of 3 custom ops. A shell script for compiling these ops is located at `tf_ops/compile_ops.sh`. Ensure `CUDA_ROOT` and `EIGEN_LIB` point to your local installations.

The ops can then be comiled from the root directory by running:

```
chmod u+x ./tf_ops/compile_ops.sh
sh ./tf_ops/compile_ops.sh
```

## Dataset generation

We provide preprocessing code for both the Stanford 3D-S dataset and ScanNet. First create a new folder in the root directory called data (`./data`).

### Stanford 3D-S dataset

Download the [S3DIS dataset](http://buildingparser.stanford.edu/dataset.html) and place the downloaded folder into `./data`. 

#### Loss network

The loss network training data consists of small patches with a random but known offset. The script for generating training data is found in `datasets/s3dis/s3dis_patches.py`. The training data generation can be customised using the `config` dictionary at the bottom of the file. We leave the deault values to those used in the paper. To change the percentage of training data used change the `train_ratio` variable, where 1. and .05 would be 100% and 5% of available training data respectively. `n_crops_per_object` and `n_rotations_per_object` are data augmentation settings. Setting these high will give more varied data samples from the available training data. `n_negative_samples` is the number of data samples with where the object is not present. We find best results are achieved when this is balanced with `n_crops_per_object`.

Once the `config` is correctly configured, you can generate the loss network train `.tfrecord` files by running:

```
python datasets/s3dis/s3dis_patches.py
```

#### Scene network

The dataset must first be processed for scene level supervision. First ensure the config file is set correctly and run:

```
python datasets/s3dis/process_scenes.py
```

The `config` dictionary should be configured to point to the dataset root directory and the processed data directory. `box_size` and `overlap` are added together to get total scene size (i.e. `box_size: (1.5, 1.5)` and `overlap: (1.5, 1.5)` would result in a scene size `(3., 3.)`. If `rotate` is set to `True`, 3 rotation augmentations will be performed at 90, 180, and 270 degrees. Finall ensure the config points to the correct folders and run:

```
python datasets/s3dis/s3dis_scenes.py
```

### ScanNet

Download [ScanNet](http://www.scan-net.org/) and place in `./data`. 

See the Loss network section for Stanford 3D-S data for more details on configs. Once configured generate patches and scenes with:

```
python datasets/scannet/scannet_patches.py
python datasets/scannet/scannet_scenes.py
```

## Training

### Loss network

Before the scene network can be trained you must first train a loss network, which in turn is used to train the scene network. Before training you must configure the `config` dictionary at the bottom of the train file `loss_network/train.py`. `train_dataset` and `test_dataset` should point to the `.tfrecord` files created in dataset generation step. `test_freq` determines how many steps a test batch is run. `log_freq` sets the frequency that the metrics are logged to tensorboard.

Once configured, training can be initalised with:

```
python loss_network/train.py
```

Logs will be written to the log directory and can be visualised in tensorboard with:

```
tensorboard --logdir=/path/to/logs
```

The best model will be saved by deafult as: `/path/to/logs/models/weight.ckpt`. This is the model to use for training the scene network.

Note: The model overwrites itself each time the mean loss passes below the previous best. Training will run indefinitely until cancelled. Once the loss curves in the tensorboard have converged simply kill the training (i.e. `ctrl + c`).

Once training is complete you can evaluate the model performance on the entire test set by configuring and running:

```
python loss_network/evaluate.py
```

### Scene network

To train the scene network point the `config` dictionary at the bottom of `scene_network/train.py` to the `.tfrecord` files generated above. `loss_weights` should also point to the trained loss network model. `loss_points` should match the number of points the loss network was trained with. To train the network using the learned loss set `loss_network_inf: True` and `unsupervised: True`. To train using the supervised chamfer loss function set `unsupervised: False`. If `loss_network_inf` is still `True` then the gradients will be updated from the supervised loss function, but the patches will still be passed through the loss network and metrics collected. This is useful for comparison. However, it is slower than using the supervised loss network by itself. Therefore, if to train supervised it is also advisable to set `loss_network_inf: False`. Once the `config` is configured correctly, start scene network training with:

```
python scene_network/train.py
```

As with the loss network, logs will be written to the `log_dir` and training can be visualised in tensorboard by running:

```
tensorboard --logdir=/path/to/logs
```

> **Note**: Supervised losses are plotted for training using the learned loss function, however, it is important to remember these have different objective functions. As such, when training using the learned loss the supervised loss curves do not necessarily indicate model performance.

## Evaluation

After successful training you can evaluate the performance of the model using the `scene_network/inference.py` script. The `config` dictionary can be used to point towards the trained scene network model. `score_thresh` is a hyperparameter which is used to determine the at which confidence the occupancy branch needs to be for to classify the prediction as a positive. Increasing this therefore improves precision at the cost of recall. For the results presented in the paper we use the default `0.9`. `iou_thresh` is the minimum required IoU for a prediction and a ground truth box such that the prediction is classified as true positive. For example if `iou_thresh=0.25` the ap score is mAP@0.25. The evaluation can be run by:

```
python scene_network/evaluation.py
```

The final results will be printed to the terminal. A `.csv` file will also be saved in the `log_dir` for the current model being evaluated with per sample results.

## Inference

We also provide a full room inference script for S3DIS dataset to run the scene network over an entire room with no redundency. This is purely for visualisation purposes and do not necessarily reflect the results reported in the paper with respect to mAP@.25. In the paper we report results on individual samples of scene network training data. To run first configure the `config` dictionary at the bottom of `scene_network/inference.py` to point to the area and room you wish to run and the trained model. Then perform inference with:

```
python scene_network/inference.py
```

