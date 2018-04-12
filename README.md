# TFRecord_Train
Training cnn model from yaml and tfrecords

1. shuffle_path.py to generate shuffled path of all images in the train folder path 
2. create_tf_record to create TFRecord file from shuffled path list
3. .yaml file contains the model structure
4. trainnew.py execute the training from yaml file 
