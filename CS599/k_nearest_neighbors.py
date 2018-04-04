'''
Neural Networks CS599
@author: Sara Marie Mc Carthy
@ID: 6096340723
'''
import numpy as np
import sys
import pickle
from sklearn.decomposition import PCA

def unpickle(pfile):
    with open(pfile, 'rb') as f:
        return pickle.load(f)

def LoadData(pfile):
    with open(pfile, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        
        images = data[b'data']
        labels = data[b'labels']
        
        reshaped_img = images.reshape([-1, 3, 1024])
        gray = np.asarray(0.299)*reshaped_img[:,0,:]+np.asarray(0.587)*reshaped_img[:,1,:]+np.asarray(0.114)*reshaped_img[:,2,:]

        return gray, labels

def getKNN(point, neighbors, K, labels):
    # dictionary mapping distances to corresponding neighbors
    distances = np.zeros(len(neighbors))#{}
    i=0
    for p2 in neighbors:
        #compute distance from point to neighbor p2
        dist = np.linalg.norm(point-p2)
        distances[i]=dist
        i+=1

    # sort the distances in ascending order    
    indices = distances.argsort()
    sorted_dist = np.sort(distances)
    sorted_neighbors = neighbors[indices]
    l=np.array(labels)
    sorted_labels = l[indices]
    KNN = sorted_neighbors[:K] # holds the k-nearest neighbors
    
    return KNN, sorted_dist[:K], sorted_labels[:K]

def getLabel(KNN, distances, labels):#train_data_dict):
    # holds the weight "number of votes" for each class label
    election = {}   # holds the weights of all the labels
    
    maxweight = 0 # holds the weight of the current winning label
    maxlabel = None # hold the current winning label
    for i in range(len(KNN)):
        label = labels[i]
        dist = distances[i]
        if label in election.keys():
            election[label] += 1/dist
        else:
            election[label] = 1/dist
        if election[label] >= maxweight:
            maxweight= election[label]
            maxlabel = label
    return maxlabel

def getDist(p1, p2):
    dist = 0
    for x in range(len(p1)):
        dist += (p1[x]-p2[x])*(p1[x]-p2[x])
    return np.sqrt(dist)      
      
if __name__ == '__main__':
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    path = sys.argv[4]
    

    images, labels = LoadData(path)
    
    #split the training and test set
    train_data = images[N:1000]
    train_labels = labels[N:1000]
    test_data = images[:N]
    test_labels = labels[:N]
    
    #do PCA
    pca = PCA(n_components=D, svd_solver='full')
    pca.fit(train_data)
    #transform the training and test sets
    pca_train = pca.transform(train_data)
    pca_test = pca.transform(test_data)
     
    for i in range(len(test_data)):
        KNN, distances, labels = getKNN(pca_test[i], pca_train, K, train_labels)
        predicted_label = getLabel(KNN, distances, labels)
        label =  test_labels[i]
        print(str(predicted_label)+" "+str(label))

    
        