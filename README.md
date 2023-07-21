# Periodic Residual Learning for Crowd Flow Forecasting (SIGSPATIAL 2022) ([official version](https://dl.acm.org/doi/10.1145/3557915.3560947) / [arXiv version](https://arxiv.org/abs/2112.06132))

*This is the implement of PRNet and its enhanced version using Python 3.8.5, Pytorch 1.7.1, tested on Ubuntu 18.04 with a GeForce RTX 2080 Ti GPU.*

## Requirements

PRNet uses the following dependencies:

* Pytorch 1.7.1 and its dependencies
* Numpy and Scipy
* CUDA 9.2 or latest version. And **cuDNN** is highly recommended


## Dataset
The datasets we use for model training is detailed in Section 6 of our manuscript.
Please download the dataset TaxiBJ-P1.zip for TaxiBJ-P1 from [Google Drive](https://drive.google.com/drive/folders/1ukHy0vsOpEn8HEkU55NdltvofZ76isB4?usp=drive_link).

Put it in folder `./datasets` and unzip it.

## Model Training
Main input arguments:
- dataset: which dataset to use
- batch_size: training batch size
- base_lr: learning rate
- prev_step: the length of observed segment
- pred_step: the length of target segment
- n_layers: the total number of SCE Blocks
- n_filters: number of filters in each SCE Block
- map_width: width of flow map
- map_height: height of flow map
- s_r: the remaining height/width for SEM module
- s_flag: enable the SEM
- c_flag: enable the clossness segment
- datapath: the path of dataset
- patience: the patience for early stop
- max_epochs: the max number of training epochs
- mode: train or test
- log_dir: directory to save the log and checkpoint

First, please export the PYTHONPATH the model directory.

Then, the following examples are conducted on dataset TaxiBJ-P1:

* Example 1 (PRNet with default settings):
```
python experiments/prnet/main.py --dataset TaxiBJ-P1 --s_flag --c_flag
```

* Example 2 (PRNet enhanced version with default settings):
```
python experiments/prnet/main.py --dataset TaxiBJ-P1 --s_flag --c_flag --x_flag
```

* Example 3 (PRNet using arbitrary settings):
```
python experiments/prnet/main.py --dataset TaxiBJ-P1 --max_epochs 200 --n_layers 9 --n_filters 64 --s_flag --c_flag
```

The trained model will be saved to './log'

## Model Test
To test above trained models, you can use the following command to run our code:
```
python experiments/prnet --dataset TaxiBJ-P1 --mode test --s_flag --c_flag
```
