# Imports packages
from utility_functions import load_checkpoint, predict

# check if path is correct to checkpoint file
try:
    # load neural network model and class to index mapping table 
    nn_model, _, class_to_idx, _ = load_checkpoint(filepath='densenet_test.pth', print_state=False)
except(NameError, FileNotFoundError, KeyError):
    print('Wrong path to checkpoint or incompatible file!')

# check if path is correct to image file
try:
    top_p, top_class = predict('flower_data/test/1/image_06760.jpg', nn_model, topk=5, gpu=True)
    print('Top classes: {}\nTop probabilities: {}'.format(top_class, top_p))
except(NameError, FileNotFoundError):
    print('Wrong path to image or incompatible file!')