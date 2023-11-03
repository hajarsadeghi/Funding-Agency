import os
import sys
import pickle
import numpy as np
import pandas as pd
from time import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import logging



logging.basicConfig(
    filename='cluster.log',  
    level=logging.INFO,      
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# default cluster is 2
# second parameter is the K value
cluster_num = 2
train_index, test_index = 10, 20
if len(sys.argv) == 4:
  cluster_num = int(sys.argv[1])
  train_index = int(sys.argv[2])
  test_index = int(sys.argv[3])
# log the cluster time
start = time()
logging.info(f'start time is: {start}')

# read original file
data_orig = pd.read_excel('Funding_Agency.xlsx')
# read the embeddings
with open('funding_agency_embeddings.pickle', 'rb') as file:
    embeddings = pickle.load(file)


train_data, train_embeddings = data_orig[:train_index], embeddings[:train_index]
test_data, test_embeddings = data_orig[train_index:test_index], embeddings[train_index:test_index]

# cluster the embeddings based on cluster_num
kmeansModel = KMeans(n_clusters=cluster_num)
kmeansModel.fit(train_embeddings)

# Predict cluster labels for the test data
test_data['Cluster_Labels'] = kmeansModel.predict(test_embeddings)
# Save the updated test data with cluster labels to a new Excel file
results_directory = 'results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
test_data.to_excel(os.path.join(results_directory, f'fa_with_labels_k{cluster_num}.xlsx'), index=False)

#plot the umap to see how close are embeddings in each cluster
def plot_umap(embeddings):
    reducer =  umap.UMAP()
    scaled_test_embed = StandardScaler().fit_transform(embeddings)
    embedding = reducer.fit_transform(scaled_test_embed)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c = np.linspace(0, 1, len(embedding)),
        cmap='rainbow',  
        s= 0.1
    )
    plt.title('UMAP of Test Data Embeddings', fontsize=20);
    plt.savefig(f'umap_embeddings_k{cluster_num}.png')

plot_umap(test_embeddings)

# log finish time
end = time()
logging.info(f'End time is: {end}')
# Calculate the time taken
time_taken = end - start
logging.info(f'Time taken: {time_taken/60:.3f} minutes')
#shutdown logging
logging.shutdown()