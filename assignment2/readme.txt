Python version used 3.6.7

There are total 5 code files. The procedure to run them to get fine and coarse classification is as follows:

For coarse classification:

1. To extract the features for the images,run 'python coarse_grained_classificaion.py' . This generates features for all images and divides them into train and validation set.

2. To train the model run 'python train.py' . This trains and saves the model for coarse classification.

For fine classification:

1. To generate the train and validation data and a dictionary for labels , run 'python fine.py'

2. To train the model run 'python resnet_bilinear_cnn.py' . This has to be done for each coarse class to generate the weights for that class.


To test after generating all the weights for fine class and model for coarse class, save all the test images in a folder and run 'python test.py'. This will generate a .txt file conataining the output in the required format.   
