# Image-Classifier
Second project on Udacity Nanodeegre program - Introduction to Python for AI Programmers

## How to use this app:

### Setting up enviroment

1. Install [Anaconda](https://www.anaconda.com)
2. Verify that conda has been properly installed by typing the command:\
 ```conda -V```\
In response, you should get a similar output:\
 ```conda 23.1.0```
3. Go to repo directory: ```cd \myownpath\Image-Classifier```
4. Create an environment from yaml file by entering the command:\
```conda env create -f torchvision_env.yml``` \
The environment file is containing pytorch library and [CUDA](https://developer.nvidia.com/cuda-downloads) drivers
5. Activate the newly created environment by typing:\
 ```conda activate torchvision_env ```


### Training neural network


#### Available options:

- Display script help:\
`-h, --help`\
Example usage: `python train.py -h`
- Select convolutional neural network model architecture\
The script now supports three different models: [AlexNet](https://paperswithcode.com/method/alexnet), [DenseNet](https://paperswithcode.com/method/densenet) and [VGG-16](https://paperswithcode.com/method/vgg-16)  
`--arch {alexnet, densenet, vgg16}`\
Example usage: \
`python train.py flower_data --arch densenet`
- Select learning rate (optimizer step size on each iteration): $\alpha$ < 1 \
Default value: $\alpha$ = 0.001\
`-lr, --learning_rate `\
Example usage: \
`python train.py flower_data -lr 0.005`
- Set the number of hidden units in classifier (default: 512)\
`-hu, --hidden_units`\
Example usage: \
`python train.py flower_data --hidden_units 256`
- Choose the number of epochs to train (default: 20)\
`-e, --epochs`\
Example usage: \
`python train.py flower_data -e 50`
- Set directory to save checkpoint file (default: cwd\checkpoint.pth)\
`-s, --save_dir`\
The file name must end with an extension `.pt` or `.pth`\
Example usage: \
`python train.py flower_data --save_dir C:\Users\Tomasz\Desktop\Image-Classifier\alexnet.pt`

- Accelerate the learning by moving computation to the GPU\
`--gpu`\
Example usage:\
 `python train.py flower_data  --gpu`
