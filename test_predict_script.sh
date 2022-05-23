# Basic tests for predict.py script

# basic usage
python predict.py flower_data/test/99/image_07833.jpg densenet_e20.pth
# check if change in number of top k is working 
python predict.py flower_data/test/99/image_07833.jpg densenet_e20.pth --top_k 3
# check if label mapping is working properly 
python predict.py flower_data/test/99/image_07833.jpg densenet_e20.pth --category_names cat_to_name.json
# check if predicting on gpu works
python predict.py flower_data/test/99/image_07833.jpg densenet_e20.pth --gpu
