# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from sklearn.cluster import KMeans

#from google.cloud import storage
from io import BytesIO
import os
import time
import glob
import cv2

########################### THIS IS A Supervised clustering example  #############################

# Image files stored in the folders
my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'), key=lambda x: int(x.split("/")[-1].split(".")[0]))
print(my_files_1)


start = time.time()

model = ResNet50(weights='imagenet', pooling=max, include_top=False)


####### GENERATING and Accumulating All FEATURES

start = time.time()

k=0
my_feature = []  # All feature is to stored here
small_file =[]
N_of_Cluster =3

for index in range(100): # Here the number 1000

        # f = cv2.imread(os.path.join(my_files_1, file))
        # img = image.load_img(f, target_size=(224, 224))
        im = cv2.imread(my_files_1[index])
        small_file.append(my_files_1[index])
        im = cv2.resize(im, (224, 224))
        cv2.imshow("Image",im)
        cv2.waitKey(1)
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features = np.array(features)
        features = np.ravel(features)
        my_feature.append(features)
        print("index ",index, "   feature_shape ",features.shape)

        #features_reduce = features.squeeze()
        #train_featues.write(' '.join(str(x) for x in features.squeeze()) + '\n')


# Build the model using the face data

labels =[]

def cluster (my_feature):

        print("\n\n clustering in Progress")

        kmeans = KMeans(n_clusters=N_of_Cluster)
        kmeans = kmeans.fit(my_feature)

        # Get cluster numbers for each face
        labels = kmeans.predict(my_feature)
        print(labels)

        return labels


# for (label, face) in zip(labels, faces):
#     face["group"] = int(label)

labels = cluster(my_feature)

def save_img_cluster (labels):

        for i in range(N_of_Cluster+1):
                try :
                        os.mkdir('./Clustered_folder/For_balck_video/'+str(i))
                except:
                        continue


        for index, lb in enumerate(labels):

            img = cv2.imread(my_files_1[index])
            write_path = './Clustered_folder/For_balck_video/' + str(lb) + '/'
            cv2.imwrite(write_path + str(index)+'.jpg',img)


save_img_cluster(labels)


end = time.time()
print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')
cv2.destroyAllWindows()
