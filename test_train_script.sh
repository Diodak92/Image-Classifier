# Basic tests for train.py script
# show program help
python train.py -h
# basic usage
python train.py flower_data
# check if function for saving checkpoint in desired location works propely
python train.py flower_data --save_dir sample_checkpoint
# check if option for select different neural network architectures works
python train.py flower_data --arch alexnet
# check if program works with different hyperparameters
python train.py flower_data -lr 0.01 --hidden_units 256 --epochs 2 --gpu
# check case for error in dataset file path
python train.py flowerData