import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
from lshash.lshash_2_py3 import LSHash

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
        self.range = 1000


    def init_lsh(self, no_bit, no_hash_per_table):

        self.lsh = LSHash(hash_size=no_bit, input_dim=self.range,
                          num_hashtables=1, num_hash_per_tables=no_hash_per_table,hash_type= 'pca_bin', storage_config=None,
                          matrices_filename=None, overwrite=False)

        print(" LSH pbject instantiated ")



    def avg_downsample(self,feature):

        k =8  # sampling factor
        f= feature.reshape(-1, k).mean(axis=1)
        return f

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


    def get_vgg_feature(self, im):

        im = cv2.resize(im, (224, 224))
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        features = np.array(features)
        features = np.ravel(features)

        return features

    def add_noise_into_img(self,img):

        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
        return img



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
            features = self.avg_downsample(features)
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

        #print (self.model.summary)

    def indexing_feature(self,feature,additional_data, indx):

        self.lsh.index(feature[0:self.range], additional_data)

        print("indexing ::", indx)
        # print("indexing ::", additional_data)
        # print("indexing ::", feature)


    def query_image(self,img_feature):

        s = time.time()
        query_result = self.lsh.query(img_feature[0:self.range], num_results=10, distance_func='euclidean')
        e = time.time()

        print(" \n query_time ", e-s, '\n')
        return query_result


    def hashing_clustered_image(self):

        for index, (f,label) in enumerate(zip(self.my_feature, self.labels)):

            self.indexing_feature(feature=f,additional_data=self.my_files_1[index].split('/')[-1]+ "  ,Folder " + str(label) ,indx = index)


    def test_blur_img(self,imgRange):

        for index in range(imgRange):

            im = cv2.imread(self.my_files_1[index])
            im = cv2.resize(im, (224, 224))
            im = self.add_noise_into_img(im)

            cv2.imshow("Image", im)
            cv2.waitKey(1)

            x = image.img_to_array(im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.model.predict(x)
            features = np.array(features)
            features = np.ravel(features)
            features = self.avg_downsample(features)

            result = self.query_image(img_feature=features)
            print(" \n index ", index ,result)


if __name__ == '__main__':


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

    svc.range = np.array(svc.my_feature).shape[1]
    svc.init_lsh(no_bit=12,no_hash_per_table=20)
    svc.hashing_clustered_image()

    ############## Test Blur Images #################

    svc.test_blur_img(imgRange)


    # for ff in svc.my_feature:
    #
    #     result=svc.query_image(img_feature=ff)
    #     print(result)



    print(np.array(svc.my_feature).shape)
    cv2.destroyAllWindows()
