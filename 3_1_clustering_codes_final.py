# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:13:18 2020

@author: nshah12
"""

###Load libraries
import concurrent.futures


import numpy as np
import pandas as pd
import time
import json


#clustering library
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score 
# from validclust import dunn
from sklearn.metrics import pairwise_distances

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import os 
from multiprocessing import Pool


# root = os.getcwd()




np.random.seed(0)

def read_data():

    ###READ model data
    model_data_df=pd.read_csv('_Data/model data/model_data_pca_TimeZoneCorrected_08192021.csv')#.sample(1000)
    model_data=model_data_df.iloc[:,1:11]

    return model_data

def get_score(cluster_number,model_data):

    # create an instance of the model, and fit the training data to it.
    kmeans = KMeans(n_clusters=cluster_number,
                    init='k-means++',
                    random_state=0).fit(model_data)

    # define the silhouette score
    sil_score_kmean = metrics.silhouette_score(model_data,
                                               kmeans.labels_,
                                               metric='euclidean')
    #    dist = pairwise_distances(pca_dataset_df)
    DB_score= davies_bouldin_score(model_data,kmeans.labels_)
    
    elbow=kmeans.inertia_
    
    result_list=['Kmeans',int(cluster_number),sil_score_kmean,DB_score,elbow,kmeans.labels_.tolist()]
    
    return result_list

#-----------------------------------------------

def main():
    start_time = time.time()
    
    model_data=read_data()#.sample(1000)
    
    print(model_data.shape)
    result_colname=['Cluster_type','Cluster_no','Silhouette_score','DB_score','Elbow','Kmeans_label_list']
    # kmeans_results=pd.DataFrame(columns=result_colname)

    
    
    
    arguments_for_pool=[[i, model_data] for i in np.arange(2,20,1)]
    num_cores=20
    
    with Pool(num_cores) as pool:
        pool_results = pool.starmap(get_score,arguments_for_pool)    
    kmeans_results=pd.DataFrame(pool_results, columns=result_colname)
    
    
    # open output file for reading
    # open output file for writing
    # with open('results\KMeans_Results_WithLabels_TimeZoneCorrected08202021.txt', 'w') as filehandle:
    #     json.dump(pool_results, filehandle)
    
    kmeans_results.to_csv(str('results\kmeans_results_all_TimeZoneCorrected_08202021_only10PCs.csv'))

    print("Execution time: ",(time.time() - start_time)/60)
    
    print(kmeans_results)
    
    
    return None

    
if __name__ == '__main__':
    main()


# # open output file for reading
# with open('results\listfile.txt', 'r') as filehandle:
#     basicList = json.load(filehandle)



# for cluster_number in np.arange(2,20,1):
# #    kmeans_summary_dict = {}

#     #---------------------------------------------------
#     #### K-means

#     # create an instance of the model, and fit the training data to it.
#     kmeans = KMeans(n_clusters=cluster_number,
#                     init='k-means++',
#                     random_state=0).fit(model_data)

#     # define the silhouette score
#     sil_score_kmean = metrics.silhouette_score(model_data,
#                                                kmeans.labels_,
#                                                metric='euclidean')
# #    dist = pairwise_distances(pca_dataset_df)
#     DB_score= davies_bouldin_score(model_data,kmeans.labels_)
    
#     result_list=['Kmeans',cluster_number,sil_score_kmean,DB_score]
    
#     kmeans_results=kmeans_results.append(pd.Series(['Kmeans',cluster_number,
#                                      sil_score_kmean,DB_score], index=result_colname), 
#         ignore_index=True)
    
    
# #    
#     print('done:',cluster_number)
    
#     # return kmeans_summary_dict


# kmeans_results.to_csv(str('results\kmeans_results_all.csv'))
# #with concurrent.futures.ThreadPoolExecutor(max_workers=400) as executor:
# #    results_kmean = executor.map(clustering_kmean, range(2, 20))



# #for i in range(2, 5):
#    clustering_kmean(i)


