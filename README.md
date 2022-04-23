# HFANet

Thanks for your interest in our work, "[HFA-Net:High Frequency Attention Siamese Network for Building Change Detection in VHR Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0031320322001984)". More detailed information can be find at DOI: 10.1016/j.patcog.2022.108717

## Citation
If you find our work useful for your research, please consider citing our paper:
```
@article{ZHENG2022108717,
title = {HFA-Net: High frequency attention siamese network for building change detection in VHR remote sensing images},
journal = {Pattern Recognition},
volume = {129},
pages = {108717},
year = {2022},
issn = {0031-3203},
}
```

## Instructions

### Datasets

Firstly, you can arrange your datasets as follows:  <br />
├── YourDataset  <br />
│ ├── train    <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
│ ├── test     <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
............   <br />
Then you can set the path to the dataset in Dataset.py. It should be noted that all the images need to be named as "number.tif", e.g., "1.tif,11.tif,111.tif".

### Hyperparameters

The batch sizes for training and testing, optimizers, learning rate and so on can be set or changed in Main.py.

### Train & Test

After the corresponding environment is installed successfully, excute the code below to train and test HFA-Net over your dataset. 

```shell

python Main.py --dataset $YourDataset

```
The evaluation metrics and the best visualized results will be saved for your convenience.

### Environment
__pytorch__ 1.8.0 with corresponding __CUDA toolkits__. <br />
__torchvision__ 0.9.0 <br />
argparse 1.4.0 <br />
opencv-python 4.5.4.58 <br />
tqdm 4.62.3 <br />

If there are unavoidable problems or inconveniences for you to directly implement HFA-Net, you can easily extract the bare untrained model of HFA-Net saved in Net.py, Attention_Module.py, High_Frequency_Module.py, Encoder.py and Decoder.py and utilize it in your work for comparison. <br />
In addition, you can directly acquire the quantitative experimental results of HFA-Net over three widely used VHR change detection data sets.

## Acknowledgement

We are grateful for outstanding contributions of three open change detection data sets, i.e., WHU-CD [1], LEVIR-CD [2], Google Data set [3]:  <br />
``` 
[1] Ji S, Wei S, Lu M. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set[J]. IEEE Transactions on Geoscience and Remote Sensing, 2018, 57(1): 574-586.  
[2] Chen H, Shi Z. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection[J]. Remote Sensing, 2020, 12(10): 1662.
[3] D. Peng, L. Bruzzone, Y. Zhang, H. Guan, H. Ding and X. Huang. SemiCDNet: A Semisupervised Convolutional Neural Network for Change Detection in High Resolution Remote-Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021, 59(7): 5891-5906.
```



