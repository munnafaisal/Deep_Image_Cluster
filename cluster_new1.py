import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
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

class SupervisedCluster():
    def __init__(self):
        self.my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.model = ResNet50(weights='imagenet', pooling=max, include_top=False)
        self.k = 0
        self.my_feature = []  # All feature is to stored here
        self.small_file = []
        self.N_of_Cluster = 3
        self.labels = []
        self.clusteredImgSavePath = './Clustered_folder/For_balck_video/'

    def cluster(self):
        print("\n\n clustering in Progress")

        kmeans = KMeans(n_clusters=self.N_of_Cluster)
        kmeans = kmeans.fit(self.my_feature)

        # Get cluster numbers for each face
        self.labels = kmeans.predict(self.my_feature)
        print(self.labels)

        #return self.labels

    def save_img_cluster(self):

        for i in range(self.N_of_Cluster + 1):
            try:
                os.mkdir(self.clusteredImgSavePath + str(i))
            except:
                continue

        for index, lb in enumerate(self.labels):
            img = cv2.imread(self.my_files_1[index])
            write_path = self.clusteredImgSavePath + str(lb) + '/'
            cv2.imwrite(write_path + str(index) + '.jpg', img)

    def main(self,imgRange):
        for index in range(imgRange):  # Here the number 100

            im = cv2.imread(self.my_files_1[index])
            self.small_file.append(self.my_files_1[index])
            im = cv2.resize(im, (224, 224))
            cv2.imshow("Image", im)
            cv2.waitKey(1)
            x = image.img_to_array(im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.model.predict(x)
            features = np.array(features)
            features = np.ravel(features)
            self.my_feature.append(features)
            print("index ", index, "   feature_shape ", features.shape)
            #return self.my_feature

    def imageRange(self):
        path, dirs, files = next(os.walk("./Cropped_Image_2/Cropped_Image"))
        file_count = len(files)
        print(file_count)
        return file_count

    def modelSelect(self,var):
        if(var=='1'):
            self.model= ResNet50(weights='imagenet', pooling=max, include_top=False)
        elif(var=='2'):
            self.model = VGG19(weights='imagenet', pooling=max, include_top=False)




if __name__ == '__main__':
    start = time.time()
    start = time.time()
    print("1:Resnet50\n2:VGG19\n3:MobilenetSSD\n")
    var=input()

    svc=SupervisedCluster()
    #imgRange=svc.imageRange()
    svc.modelSelect(var)
    imgRange=100
    svc.main(imgRange)

    svc.cluster()
    svc.save_img_cluster()
    end = time.time()
    print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')
    cv2.destroyAllWindows()
