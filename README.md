# Image-Classifier
Project created as partof the Udacity Nanodegree program "Introduction to Python for AI Programmers"

## How to use this app:

### Set up enviroment

1. Install [Anaconda](https://www.anaconda.com)
2. Verify that Conda has been properly installed by typing the command:\
 ```conda -V```\
In response, you should get a similar output:\
 ```conda 23.1.0```
3. Go to repo directory: ```cd \myownpath\Image-Classifier```
4. Create a new environment from a yaml file by entering the command:\
```conda env create -f torchvision_env.yml``` \
The environment file contains [pytorch](https://pytorch.org) library and [CUDA](https://developer.nvidia.com/cuda-downloads) drivers.
5. Activate the newly created environment by typing:\
 ```conda activate torchvision_env ```


## Training the neural network

In order to train an image classifier use the `train.py`
script on a particular image dataset. The script takes the image dataset folder as an input and returns a trained neural network as a checkpoint file, which also contains information about the training state and model performance.\
The script currently supports three different convolutional neural network models: [AlexNet](https://paperswithcode.com/method/alexnet), [DenseNet](https://paperswithcode.com/method/densenet) and [VGG-16](https://paperswithcode.com/method/vgg-16).
### Features:
- The script automatically detects the number of classes and rebuilds the network classfier.
- Various options for training are available. 


### Positional arguments:
- Train image dataset directory:\
  `dir`\
  File structure has to be formated as follows:
- `image_folder\train\`
  - `\category_1\`
    - `\image_1.jpeg`
    - `\image_2.jpeg`
    - `...`
    - `\image_n.jpeg`
  - `\category_2\`
    - `...`
  - `\category_n\`
    - `...`
- `image_folder\test\`
  - `\category_1\`
    - `\image_1.jpeg`
    - `\image_2.jpeg`
    - `...`
    - `\image_n.jpeg`
  - `\category_2\`
    - `...`
  - `\category_n\`
    - `...`
- `image_folder\valid\`
  - `\category_1\`
    - `\image_1.jpeg`
    - `\image_2.jpeg`
    - `...`
    - `\image_n.jpeg`
  - `\category_2\`
    - `...`
  - `\category_n\`
    - `...`

Use example:\
`python train.py flower_data`

### Optional arguments:

- Display script help:\
`-h, --help`\
Example usage:\
`python train.py -h`

- Select convolutional neural network model architecture: [AlexNet](https://paperswithcode.com/method/alexnet), [DenseNet](https://paperswithcode.com/method/densenet) or [VGG-16](https://paperswithcode.com/method/vgg-16)  
`--arch {alexnet, densenet, vgg16}`\
Use example: \
`python train.py flower_data --arch densenet`

- Select learning rate (optimizers step size on each iteration): $\alpha$ < 1 \
Default value: $\alpha$ = 0.001\
`-lr, --learning_rate `\
Use example: \
`python train.py flower_data -lr 0.005`

- Set the number of hidden units in a classifier (default: 512)\
`-hu, --hidden_units`\
Use example: \
`python train.py flower_data --hidden_units 256`

- Choose the number of epochs to train (default: 20)\
`-e, --epochs`\
Example usage: \
`python train.py flower_data -e 50`

- Set directory to save checkpoint file (default: cwd\checkpoint.pth)\
`-s, --save_dir`\
The file name must end with the extension `.pt` or `.pth`\
Use example: \
`python train.py flower_data --save_dir C:\Users\Tomasz\Desktop\Image-Classifier\alexnet.pt`

- Accelerate learning by moving computation to the GPU\
`--gpu`\
Example usage:\
 `python train.py flower_data  --gpu`
