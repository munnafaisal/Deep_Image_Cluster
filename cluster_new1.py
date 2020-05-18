import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from lshash.lshash_2_py3 import LSHash
from skimage.util import random_noise
from sklearn.neighbors import NearestNeighbors



#from google.cloud import storage
from io import BytesIO
import os
import time
import glob
import cv2

#dirs = sorted(glob.glob('./Clustered_folder/For_balck_video/*/'))
#files = sorted(glob.glob(dirs[0]+'*.jpg'))

class SupervisedCluster():

    def __init__(self):

        #self.my_files_1 = sorted(glob.glob('./Cropped_Image_2/Cropped_Image/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.my_files_1 = sorted(glob.glob('./temp/person/*.jpg'),key=lambda x: int(x.split("/")[-1].split(".")[0]))
        #self.model = ResNet50(weights='imagenet', pooling=max, include_top=False)
        self.k = 0
        self.my_feature = []  # All feature is to stored here
        self.small_file = []
        self.N_of_Cluster = 4
        self.N_of_sub_Cluster = 5
        self.labels = []
        self.clusteredImgSavePath = './Clustered_folder/For_balck_video/'
        self.range = 1000

    def sub_cluster_features(self):

        self.cluster_features = []
        self.labels = np.array(self.labels)
        self.my_feature = np.array(self.my_feature)

        for n in range(0,self.N_of_Cluster):

            k = np.where(self.labels==n)[0]
            f = np.take(self.my_feature,k,axis= 0)
            self.cluster_features.append(f)

            print("\n cluster no ", n)
            print(",\n features \n ",f.shape )

        return self.cluster_features



    def init_lsh(self, no_bit, no_hash_per_table):

        self.lsh = LSHash(hash_size=no_bit, input_dim=self.range,
                          num_hashtables=1, num_hash_per_tables=no_hash_per_table,hash_type= 'pca_bin', storage_config=None,
                          matrices_filename=None, overwrite=False)

        print(" LSH object instantiated ")

    def normalise_block_embed(self,embed):

        k = 4  # number of block

        mres = np.array(embed).ravel().reshape(-1,k)
        mres = mres / np.linalg.norm(mres, ord=2, axis=0, keepdims=True)
        mres = mres.ravel()

        return mres

    def histogram_equalization(self,img_in):

        # segregate color streams
        b, g, r = cv2.split(img_in)
        h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
        h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
        h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
        # calculate cdf
        cdf_b = np.cumsum(h_b)
        cdf_g = np.cumsum(h_g)
        cdf_r = np.cumsum(h_r)

        # mask all pixels with value=0 and replace it with mean of the pixel values
        cdf_m_b = np.ma.masked_equal(cdf_b, 0)
        cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
        cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

        cdf_m_g = np.ma.masked_equal(cdf_g, 0)
        cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
        cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

        cdf_m_r = np.ma.masked_equal(cdf_r, 0)
        cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
        cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
        # merge the images in the three channels
        img_b = cdf_final_b[b]
        img_g = cdf_final_g[g]
        img_r = cdf_final_r[r]

        img_out = cv2.merge((img_b, img_g, img_r))

        # validation
        # equ_b = cv2.equalizeHist(b)
        # equ_g = cv2.equalizeHist(g)
        # equ_r = cv2.equalizeHist(r)
        # equ = cv2.merge((equ_b, equ_g, equ_r))
        # print(equ)
        # cv2.imwrite('output_name.png', equ)


        return img_out

    def avg_downsample(self,feature):

        k =8  # sampling factor
        f= feature.reshape(-1, k).max(axis=1)
        return f

    def get_segmented_data(self,feature, st,en):

        f= feature[st:en]
        return f

    def get_keys(self):

        for table in self.lsh.hash_tables:
            keys = list(table.keys())

            for index, key in enumerate(keys):
                ar = [int(x) for x in list(key)]
                keys[index] = ar


        return keys

    def build_NN(self,keys,n_of_neighbour):

        nbrs = NearestNeighbors(n_neighbors=n_of_neighbour, algorithm='ball_tree').fit(keys)
        self.nbrs = nbrs

    def cluster(self):

        print("\n\n clustering in Progress")
        self.N_of_Cluster = np.int(self.N_of_Cluster)
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

    def add_salt_pepper_noise(self,img):

        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(img, mode='s&p', amount=0.05)

        # The above function returns a floating-point image
        # on the range [0, 1], thus we changed it to 'uint8'
        # and from [0,255]

        noise_img = np.array(255 * noise_img, dtype='uint8')

        return noise_img


    def main(self,imgRange):

        for index in range(imgRange):  # Here the number 100

            im = cv2.imread(self.my_files_1[index])
            self.small_file.append(self.my_files_1[index])
            im = cv2.resize(im, (224, 224))
            #im = self.histogram_equalization(img_in= im)
            cv2.imshow("Image", im)
            cv2.waitKey(1)

            features = self.get_vgg_feature(im)
            features = self.avg_downsample(features)
            features = self.normalise_block_embed(features)

            #features = self.get_segmented_data(features, st=32000, en=64000)
            self.my_feature.append(features)

            print("index ", index, "   feature_shape ", features.shape)
            #return self.my_feature

        self.range = features.shape

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
        query_result = self.lsh.query(img_feature[0:self.range], num_results=10, distance_func='normalised_euclidean')
        e = time.time()

        print(" \n query_time ", e-s, '\n')
        return query_result


    def hashing_clustered_image(self):

        for index, (f,label) in enumerate(zip(self.my_feature, self.labels)):

            self.indexing_feature(feature=f,additional_data=(self.my_files_1[index].split('/')[-1]+ "  : Folder " + str(label),np.var(f)) ,indx = index)


    def test_blur_img(self,imgRange):

        for index in range(imgRange):

            im = cv2.imread(self.my_files_1[index])
            im = cv2.resize(im, (224, 224))
            #im = self.histogram_equalization(img_in= im)

            #im = self.add_noise_into_img(im)
            im = self.add_salt_pepper_noise(im)

            cv2.imshow("Image", im)
            cv2.waitKey(1)

            features = self.get_vgg_feature(im)
            features = self.avg_downsample(features)
            features = self.normalise_block_embed(features)

            print(features.shape)
            #features = self.get_segmented_data(features, st=32000, en=64000)

            result = self.query_image(img_feature=features)

            print (len(result))
            print(" \n index ", index ,"\n",result)


    def kmean_unsupervised(self,range_cls,features):

        n_test = [n for n in range(2,range_cls)]
        s_score = []


        for n in n_test :

            #clusterer = KMeans(n_clusters=n, random_state=10)

            clusterer = AgglomerativeClustering(n_clusters=n, linkage="average")

            cluster_labels = clusterer.fit_predict(features)


            #print("\n Cluster labels :: ", cluster_labels)
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters


            silhouette_avg = silhouette_score(features, cluster_labels)
            s_score.append(silhouette_avg)

            print("For n_clusters =", n,
                  "The average silhouette_score is :", silhouette_avg)


        s_score = np.array(s_score)
        optimal_cluster = np.where(s_score == np.max(s_score))[0]+2
        print("\n optimal_cluster :: ", optimal_cluster)
        print(" \n\n ")

        return optimal_cluster




if __name__ == '__main__':


    start = time.time()
    print("1:Resnet50\n2:VGG19\n3:MobilenetSSD\n")
    var=input()

    ########### Instantiate Cluster Ibject ##########

    svc=SupervisedCluster()
    #imgRange=svc.imageRange()

    ############ Select Pretrained Object Detection Model #############

    svc.modelSelect(var)

    ################## Number of Image to be read from Image ##############

    imgRange=10000

    ############### Get Feature From Images ###############

    svc.main(imgRange)



        ########### Unsupervised Clustering ####################

################## Get optimal number of parent cluster using silhouette score  ###############

    svc.N_of_Cluster = svc.kmean_unsupervised(range_cls=3,features= svc.my_feature)

    ###################### Initiate Clustering Process ########################

    svc.cluster()

    #################### Save Images in the Parent Cluster  ##################

    svc.save_img_cluster()
    end = time.time()
    print('\n\n time spend: ', (end - start) / 60, ' minutes \n\n')


    #################### Initiate LSH Hashing ####################

    svc.range = np.array(svc.my_feature).shape[1]
    svc.init_lsh(no_bit=128,no_hash_per_table=5)


################## Get Clustered Features ####################

    svc.sub_cluster_features()

################## Get optimal number of child cluster for each parent cluster using silhouette score  ###############

    # for k in range(0,svc.N_of_Cluster):
    #
    #     print (" \n \n sub_cluster :: ", k )
    #     svc.kmean_unsupervised(range_cls=5,features= svc.cluster_features[k])


################# Start Hashing Clustered Images ######################################

    svc.hashing_clustered_image()


    # dict_keys = svc.get_keys()
    #
    # svc.build_NN(dict_keys,5)
    #
    # query_point = np.array(dict_keys[1]).reshape(1,-1)
    #
    # distances, indices = svc.nbrs.kneighbors(query_point,10)
    #
    # print("\n\n ",distances," \n\n ",indices)
    # print("query \n",dict_keys[1], "\n")
    # for inddice in indices[0]:
    #     print(dict_keys[inddice])

    #print(" \n ",dict_keys)
    svc.lsh.hash_keys = svc.lsh.get_keys()
    svc.lsh.build_NN(svc.lsh.hash_keys,100)

################## Test Blur Images #################

    svc.test_blur_img(imgRange)



    #print(np.array(svc.my_feature).shape)
    cv2.destroyAllWindows()
