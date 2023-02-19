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
- Set directory to save checkpoint file\
`-s, --save_dir`\
The file name must end with an extension `.pt` or `.pth`\
Example usage: \
`python train.py flower_data  C:\Users\Tomasz\Desktop\Image-Classifier\alexnet.pt`

- Accelerate the learning by moving computation to the GPU\
`--gpu`\
Example usage:\
 `python train.py flower_data  --gpu`
