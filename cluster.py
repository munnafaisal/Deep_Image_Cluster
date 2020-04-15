import tensorflow
import keras
from keras import Sequential
from keras.applications import ResNet50
from keras_applications import vgg16
import glob
import json
import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input
#img_files = sorted(glob.glob('Images/*.jpg'), key=lambda x: int(x.split("/")[1].split(".")[0]))

my_files_1 = sorted(glob.glob('../Images/*.jpg'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
print(my_files_1)


print("printing Image Files",my_files_1)


resnet_weights_path = 'Models/ResNet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

def extract_vector(img_files):

    resnet_feature_list = []

    for im in img_files:

        im = cv2.imread(im)
        im = cv2.resize(im,(224,224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        print(resnet_feature)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list)

extract_vector(my_files_1)