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
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')
import logging

# create a log file
logging.basicConfig(
    filename='cluster.log',  
    level=logging.INFO,      
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# log the cluster time
start = time()
logging.info(f'start time is: {start}')

# create result folder
results_directory = 'results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
# read original file
data_orig = pd.read_excel('data/Funding_Agency.xlsx')
# read the embeddings
with open('data/funding_agency_embeddings.pickle', 'rb') as file:
    embeddings = pickle.load(file)

#plot the umap to see how close are embeddings in each cluster
def plot_umap(embeddings, titles):
    reducer =  umap.UMAP()
    scaled_test_embed = StandardScaler().fit_transform(embeddings)
    embedding = reducer.fit_transform(scaled_test_embed)
    # create the scatter plot
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c = np.linspace(0, 1, len(embedding)),
        cmap='rainbow',  
        s= 0.1
    )
    plt.title('UMAP of Test Data Embeddings', fontsize=20);
    plt.savefig(os.path.join(results_directory, 'umap_embeddings.png'))

    # Create an interractive Plotly figure with the scatter plot
    fig = go.Figure(data=go.Scatter(x=embedding[:, 0], y=embedding[:, 1],
                                    text=titles,  # Display titles when hovering
                                    mode='markers'))

    # Customize the appearance and interactivity of the plot
    fig.update_layout(title="Funding Agency Interactive UMAP Visualization",
                    xaxis_title="UMAP Dimension 1",
                    yaxis_title="UMAP Dimension 2",
                    hovertext=titles,
                    hovermode='closest') 

    # Save the interactive plot as an HTML file
    fig.write_html(os.path.join(results_directory, 'interactive_umap.html'))

# Plot embeddings along with titles with umap
plot_umap(embeddings, data_orig['Funding_Agency'])


# first parameter is k
# second parameter is number of training datasets
# third parameter is number of testing datasets
cluster_num = 2
train_index, test_index = 10, 20
if len(sys.argv) == 4:
  cluster_num = int(sys.argv[1])
  train_index = int(sys.argv[2])
  test_index = int(sys.argv[3])

train_data, train_embeddings = data_orig[:train_index], embeddings[:train_index]
test_data, test_embeddings = data_orig[train_index:test_index], embeddings[train_index:test_index]

# cluster the embeddings based on cluster_num
kmeansModel = KMeans(n_clusters=cluster_num)
kmeansModel.fit(train_embeddings)

# Predict cluster labels for the test data
test_data['Cluster_Labels'] = kmeansModel.predict(test_embeddings)
# Save the updated test data with cluster labels to a new Excel file
test_data.to_excel(os.path.join(results_directory, f'fa_with_labels_k{cluster_num}.xlsx'), index=False)

# log finish time
end = time()
logging.info(f'End time is: {end}')
# Calculate the time taken
time_taken = end - start
logging.info(f'Time taken: {time_taken/60:.3f} minutes')
#shutdown logging
logging.shutdown()