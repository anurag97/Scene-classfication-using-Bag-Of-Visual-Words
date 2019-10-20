import numpy as np
import os
from sklearn import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.cluster import KMeans
# defining feature extractor that we want to use
extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    #print (keypoints)
    return keypoints, descriptors
def sift_kmeans():
    labels=['coast','highway','forest','inside_city','mountain','opencountry','street','tallbuilding']
    sift_keypoints=[]
    for label in labels:
        path='train/'+label
        for imgfile in os.listdir(path):  
            img = cv2.imread(os.path.join(path,imgfile),1)
            kp,des = features(img,extractor)
            sift_keypoints.append(des)
    sift_keypoints=np.asarray(sift_keypoints)
    sift_keypoints=np.concatenate(sift_keypoints, axis=0)
        #with the descriptors detected, lets clusterize them
    print("Training kmeans") 
    for num_cluster in range(100,500,100):
        print("No. of cluster = "+str(num_cluster))   
        kmeans = MiniBatchKMeans(n_clusters=num_cluster,random_state=0,init_size=int(num_cluster*1.2)).fit(sift_keypoints)
        print("Done Kmeans")
        pkl_filename = "pickle_model"+str(num_cluster)+".pkl"
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(kmeans,pkl_file)
    #return the learned model
def histogram_test(model,num_cluster):
    feature_vectors=[]
    class_vectors=[]
    labels=['coast','highway','forest','inside_city','mountain','opencountry','street','tallbuilding']
    for label in labels:
        print("Testing")
        path='test/'+label
        print(label)
        # dir_hist=os.path.join('hist',label)
        # if os.path.isdir(dir_hist)==False:
        #     os.makedirs(dir_hist)
        for imgfile in os.listdir(path):    
            img = cv2.imread(os.path.join(path,imgfile),1)
            kp,des = features(img,extractor)
            predict_kmeans=model.predict(des)
            # print(predict_kmeans)
            #calculates the histogram
            hist=[0 for m in range(0,num_cluster)]
            for f in predict_kmeans:
                hist[f]+=1
            # hist, bin_edges=np.histogram(predict_kmeans,bins=num_cluster)
            # n, bins, patches = plt.hist(hist, bin_edges, facecolor='blue', alpha=0.5)
            # print(dir_hist+'/'+imgfile[:-3]+'png')
            # plt.savefig(dir_hist+'/'+imgfile[:-3]+'png')
            feature_vectors.append(hist)
            class_vectors.append(label)
    feature_vectors=np.asarray(feature_vectors)
    class_vectors=np.asarray(class_vectors)
    return feature_vectors,class_vectors

def histogram(model,num_cluster):
    feature_vectors=[]
    class_vectors=[]
    labels=['coast','highway','forest','inside_city','mountain','opencountry','street','tallbuilding']
    for label in labels:
        path='train/'+label
        print(label)
        # dir_hist=os.path.join('hist',label)
        # if os.path.isdir(dir_hist)==False:
        #     os.makedirs(dir_hist)
        for imgfile in os.listdir(path):    
            img = cv2.imread(os.path.join(path,imgfile),1)
            kp,des = features(img,extractor)
            predict_kmeans=model.predict(des)
            # print(predict_kmeans)
            # print(predict_kmeans)
            #calculates the histogram
            hist=[0 for m in range(0,num_cluster)]
            for f in predict_kmeans:
                hist[f]+=1
            # hist, bin_edges=np.histogram(np.array(predict_kmeans),bins=num_cluster)
            # print(hist)
            # print(bin_edges)
            # n, bins, patches = plt.hist(hist, bin_edges, facecolor='blue', alpha=0.5)
            # print(dir_hist+'/'+imgfile[:-3]+'png')
            # plt.savefig(dir_hist+'/'+imgfile[:-3]+'png')
            feature_vectors.append(hist)
            class_vectors.append(label)
    feature_vectors=np.asarray(feature_vectors)
    

    class_vectors=np.asarray(class_vectors)
    return feature_vectors,class_vectors

#print(desc)
#img2 = cv2.drawKeypoints(img,kp,None)
#img3 = cv2.drawKeypoints(img1,kp1,None)
#cv2.imshow('photo',img2)
#cv2.imshow('photo1',img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
if __name__ == "__main__":
    sift_kmeans()
    # for i in range(100,500,100):
        


    #     filename="pickle_model"+str(i)+".pkl"
    #     model=pickle.load(open(filename, 'rb'))
    #     # print (len(model.cluster_centers_))
    #     # for m in model.cluster_centers_:
    #     #     print(len(m))
            
    #     # # print(len(model))

    #     # break
    #     train_ft,train_label=histogram(model,i)
        
    #     le = preprocessing.LabelEncoder()
    #     train_enc_label=le.fit_transform(list(train_label))
    #     # print(enc_label)
    #     test_ft,test_label=histogram_test(model,i)
    #     le1 = preprocessing.LabelEncoder()
    #     test_enc_label=le1.fit_transform(list(test_label))

    #     error=[]
    #     for j in range(5, 45):
    #         knn = KNeighborsClassifier(n_neighbors=j)
    #         knn.fit(list(train_ft), train_enc_label)
    #         pred_i = knn.predict(list(test_ft))

    #         print(confusion_matrix(test_enc_label, pred_i))
    #         print(classification_report(test_enc_label, pred_i))


    #         error.append(np.mean(pred_i != test_enc_label))


    #     plt.figure(figsize=(12, 6))
    #     plt.plot(range(5, 45), error, color='red', linestyle='dashed', marker='o',
    #          markerfacecolor='blue', markersize=10)
    #     plt.title('Error Rate K Value')
    #     plt.xlabel('K Value')
    #     plt.ylabel('Mean Error')
    #     plt.savefig("Error_for_"+str(i)+"words.png")