# Imports packages
from menu import get_input_args_predict
from utility_functions import load_checkpoint, predict, map_indexes

# get user input arguments from command line
input_args = get_input_args_predict()

# check if path is correct to checkpoint file
try:
    # load neural network model and class to index mapping table 
    nn_model, _, class_to_idx, _ = load_checkpoint(input_args.checkpoint, print_state=False)
    try:
        cat_to_name = map_indexes(input_args.category_names)
    except(NameError, FileNotFoundError, KeyError):
        print('Wrong path or incompatible JSON file!')

except(NameError, FileNotFoundError, KeyError):
    print('Wrong path to checkpoint or incompatible file!')

# check if path is correct to image file
try:
    top_p, top_class = predict(input_args.category_names, nn_model, input_args.top_k, gpu=input_args.gpu)
    idx_to_class = {i:k for k, i in class_to_idx.items()}
    labels = [cat_to_name[idx_to_class[top_class[i]]] for i in range(len(top_class))]
    print(labels)
    print('Top classes: {}\nTop probabilities: {}'.format(top_class, top_p))
except(NameError, FileNotFoundError):
    print('Wrong path to image or incompatible file!')
