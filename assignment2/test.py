import numpy as np
from PIL import Image
from keras import Model
import os
import keras
import pickle
from classification_models.resnet import ResNet18, preprocess_input
from keras.models import load_model

with open("dict_coarse.pkl" , "rb") as f:
	dict_coarse = pickle.load(f)

with open("dict_fine.pkl" , "rb") as f:
	dict_fine = pickle.load(f)

f = open("output.txt" , "w")

num_classes = 11
decay_learning_rate = 1e-8

def outer_product(x):
    return keras.backend.batch_dot(
                x[0]
                , x[1]
                , axes=[1,1]
            ) / x[0].get_shape().as_list()[1] 

def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def L2_norm(x, axis=-1):
    return keras.backend.l2_normalize(x, axis=axis)

model = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=11 , include_top = False)

x_detector = model.layers[-1].output
shape_detector = model.layers[-1].output_shape
shape_extractor = shape_detector

#print(shape_extractor)

x_extractor = x_detector
#print(x_extractor)

x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2] , shape_detector[-1]])(x_detector)

#print("After reshape",x_detector.shape)

x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2] , shape_extractor[-1]])(x_extractor)

#print("After reshape",x_extractor.shape)

x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])

#print("Outer product",x.shape)

x = keras.layers.Reshape([shape_detector[-1]*shape_extractor[-1]])(x)

#print("Reshape",x.shape)

x = keras.layers.Lambda(signed_sqrt)(x)

#print("Signed sqrt",x.shape)

x = keras.layers.Lambda(L2_norm)(x)

#print("l2 norm",x.shape)

x = keras.layers.Dense(units=num_classes,kernel_regularizer=keras.regularizers.l2(decay_learning_rate)
  ,kernel_initializer="glorot_normal")(x)

#print("Dense",x.shape)

tensor_prediction = keras.layers.Activation("softmax")(x)

#print("predictions",tensor_prediction.shape)  

model_bcnn = Model(inputs = model.input, outputs = [tensor_prediction])

model1 = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=11 , include_top = False)
layer_name = 'relu1'
intermidiate_layer_model = Model(inputs = model1.input , outputs = model1.get_layer(layer_name).output)

coarse_model = load_model('coarse_model.h5')

folder = '/home/aman9875/Documents/assignment2/test_data/'
for file in os.listdir(folder):
	img = Image.open(os.path.join(folder,file))
	img = img.resize((224,224))
	img = np.asarray(img)
	img = np.reshape(img,(1,224,224,3))
	intermidiate_layer_output = intermidiate_layer_model.predict(img)
	intermidiate_layer_output = np.reshape(intermidiate_layer_output,(1,1,7,7,512))
	t = coarse_model.predict(intermidiate_layer_output)
	t = np.reshape(t,(t.shape[1]))
	max_ = 0
	max_id = -1
	j = 0
	for i in t:
		if i > max_:
			max_id = j
			max_ = i
		j += 1
	coarse_class = dict_coarse[max_id]

	if coarse_class == "aircrafts":
		model_bcnn.load_weights("aircrafts_weight.h5")
	elif coarse_class == "birds_":
		model_bcnn.load_weights("birds_weight.h5")
	elif coarse_class == "cars":
		model_bcnn.load_weights("cars_weight.h5")
	elif coarse_class == "dogs_":
		model_bcnn.load_weights("dogs_weight.h5")
	elif coarse_class == "flowers_":
		model_bcnn.load_weights("flowers_weight.h5")

	t = model_bcnn.predict(img)
	t = np.reshape(t,(t.shape[1]))
	label = dict_fine[coarse_class]
	max_ = 0
	max_id = -1
	j = 0 
	for i in t:
		if i > max_:
			max_id = j
			max_ = i
		j += 1
	fine_class = label[max_id]
	coarse_class = coarse_class + "_"
	#print("%s %s_%s" %(file,coarse_class,fine_class))
	coarse_class,_ = coarse_class.split('_' , 1)
	f.write("%s %s %s@%s\n" %(file,coarse_class,coarse_class,fine_class))
	print("%s %s %s@%s" %(file,coarse_class,coarse_class,fine_class))