# Imports packages
from menu import get_input_args_predict
from utility_functions import load_checkpoint, predict, map_indexes

# get user input arguments from command line
input_args = get_input_args_predict()

# check if path is correct to checkpoint file
try:
    # load neural network model and class to index mapping table 
    nn_model, _, class_to_idx, _ = load_checkpoint(input_args.checkpoint, print_state=False)
except(NameError, FileNotFoundError, KeyError):
    print("Wrong path to 'checkpoint.pth' or incompatible file!")

else:
    # check if path is correct to image file
    try:
        # predict top classes with probabilities
        top_p, top_class = predict(input_args.dir, nn_model, input_args.top_k, gpu=input_args.gpu)
        
        # check if the input argument for loading the JSON file has been entered
        if input_args.category_names:
            # check if path is correct to the JSON file
            try:
                # map indexes to classes
                cat_to_name = map_indexes(input_args.category_names)
                idx_to_class = {i:k for k, i in class_to_idx.items()}
                top_class = [cat_to_name[idx_to_class[top_class[i]]] for i in range(len(top_class))]
            except(NameError, FileNotFoundError, KeyError):
                print("Wrong path to 'mapping_index_to_names.json' or incompatible file!")
        
        # print results
        print('The file photo: {} probably contains category: {}'.format(input_args.dir, top_class[0]))
        print('Top classes: {}\nTop probabilities: {}\n'.format(top_class, ["{0:0.2}".format(i) for i in top_p]))
    except(NameError, FileNotFoundError):
        print('Wrong path to image or file incompatible!')
