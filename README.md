# HFANet

Thanks for your interest in our work, HFA-Net:High Frequency Attention Siamese Network for Building Change Detection in VHR Remote Sensing Images. More detailed information can be find at DOI: 10.1016/j.patcog.2022.108717

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

If there are unavoidable problems and inconveniences for you to directly implement HFA-Net, you can easily extract the bare untrained model of HFA-Net saved in Net.py, Attention_Module.py, High_Frequency_Module.py, Encoder.py and Decoder.py and utilize it in your work for comparison. <br />
In addition, you can directly acquire the experimental results of HFA-Net over WHU-CD, LEVIR-CD, and Google Dataset at DOI: 10.1016/j.patcog.2022.108717


