#!/usr/bin python3
import os
import sys
import pickle
from time import time

from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
import logging

# create result folder
results_directory = 'results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

k1, k2 = 10, 100
startIndex, endIndex = 0, 500
if len(sys.argv) == 5:
  k1, k2 = int(sys.argv[1]), int(sys.argv[2])
  startIndex, endIndex = int(sys.argv[3]), int(sys.argv[4])

logging.basicConfig(
    filename='funding_agency.log', 
    level=logging.INFO,       
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# read the embeddings
with open('data/funding_agency_embeddings.pickle', 'rb') as file:
    embeddings = pickle.load(file)

embeddings = embeddings[startIndex:endIndex]

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

scaled_embeddings = StandardScaler().fit_transform(embeddings)
# Measure time
start = time()
logging.info(f'start time is: {start}')

# KMeans, the Elbow Method
def elbow_method(embeds, k1, k2, batch_size=1):
  k_range = list(range(k1, k2, batch_size))
  # distortions = []
  inertias = []
  
  for k in k_range:
    # Building and fitting the model
    kmeansModel = KMeans(n_clusters=k)
    kmeansModel.fit(embeds)

    # distortions.append(sum(np.min(cdist(embeds, kmeansModel.cluster_centers_,
    #                                     'euclidean'), axis=1)) / embeds.shape[0])
    inertias.append(kmeansModel.inertia_)

  return k_range, inertias

# def plot_distortion(K_range, distortions):
#   # plot elbow method with distortions
#   plt.plot(K_range, distortions, 'bx-')
#   plt.xlabel('Values of K')
#   plt.ylabel('Within Cluster SSE (aka Distortion)')
#   plt.title('The Elbow Method using Distortion')
#   plt.savefig(os.path.join(results_directory,'distortion_plot_k_'+str(k1)+'-'+str(k2)+'index'+str(startIndex)+'-'+str(endIndex)+'.png'))

def plot_inertia(K_range, inertia):
  # plot elbow method with inertias
  plt.plot(K_range, inertia, 'bx-')
  plt.xlabel('Values of K')
  plt.ylabel('Sum of Squared Distances (Inertia)')
  plt.title('The Elbow Method using Inertia')
  plt.savefig(os.path.join(results_directory,'inertia_plot_k_'+str(k1)+'-'+str(k2)+'index'+str(startIndex)+'-'+str(endIndex)+'.png'))

K, inertia_s = elbow_method(scaled_embeddings, k1, k2)
# plot_distortion(K, distor_s)
plot_inertia(K, inertia_s)

end = time()
logging.info(f'End time is: {end}')
# Calculate the time taken for UMAP
time_taken = end - start
logging.info(f'Time taken: {time_taken/60:.3f} minutes')
#shutdown logging
logging.shutdown()
