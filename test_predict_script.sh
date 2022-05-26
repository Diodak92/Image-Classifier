# Basic tests for predict.py script
# show program help
python predict.py -h
# basic usage
python predict.py flower_data/test/99/image_07854.jpg densenet_e20.pth
# check if change in number of top k is working 
python predict.py flower_data/test/99/image_07854.jpg densenet_e20.pth --top_k 3
# check if label mapping is working properly 
python predict.py flower_data/test/99/image_07854.jpg densenet_e20.pth --category_names cat_to_name.json
# check if predicting on gpu works
python predict.py flower_data/test/99/image_07854.jpg densenet_e20.pth --gpu
# check case for error in image file path
python predict.py flower_data/test/99/some_image.jpg densenet_e20.pth
# check case for error in model file path
python predict.py flower_data/test/99/image_07854.jpg googlenet_e20.pth
# check case for error in cat to name JSON file path
python predict.py flower_data/test/99/image_07854.jpg densenet_e20.pth --category_names my_own_cat_to_name.json
