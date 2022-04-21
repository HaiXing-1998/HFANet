# HFANet

Thanks for your interest in our work. This is a Siamese CNN for building change detection in VHR remote sensed imagery. More detailed information can be find at DOI: 10.1016/j.patcog.2022.108717

## Instructions

### Datasets

Firstly, you can arrange your datasets as follows:  <br />
''' <br />
├── dataset_1  <br />
│ ├── train    <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
│ ├── test     <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
├── dataset_2  <br />
│ ├── train    <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
│ ├── test     <br />
│ │ ├── A      <br />
│ │ ├── B      <br />
│ │ └── label  <br />
............   <br />
''' <br />
Then you can set the path to the dataset in Dataset.py. It should be noted that all the images need to be named as "number.tif", e.g., "1.tif,11.tif,111.tif".

### Hyperparameters

The batch sizes for training and testing, optimizers, learning rate and so on can be set or changed in Main.py.

### Train & Test

After the corresponding environment is installed successfully, excute the code below to train and test HFA-Net over your dataset. 

```shell

python Main.py --dataset $YourDataset

```


