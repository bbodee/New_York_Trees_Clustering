import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

"this is the link to the dataset that was used for this clustering algorithm"
# https://data.cityofnewyork.us/browse?q=2015%20Street%20Trees&sortBy=relevance

trees_inital_df = pd.read_csv("2015_Street_Tree_Census_-_Tree_Data.csv")

"this creates the DataFrame we want to do Kmeans clustering to, I wanted to"
"remove the extraneous data."
trees_x = trees_inital_df.loc[:,["tree_id","tree_dbh"]]

"this is commented out because the original CSV does not have commas in it,"
"but the original csv I worked on did have them"
# trees_x["tree_id"] = trees_x["tree_id"].str.replace(",","").astype("int64")


'running kmeans algorithm.  note that it is defaulting to k-means++'
kmeans = KMeans(n_clusters=4)
kmeans.fit(trees_x)
labels = kmeans.predict(trees_x)

'generating a DataFrame that saves the cluster information'
cluster_map_master = pd.DataFrame()
cluster_map_master["trees_id"] = trees_inital_df.loc[:, "tree_id"]
cluster_map_master['tree_dbh'] = trees_inital_df.loc[:,'tree_dbh']
cluster_map_master["cluster"] = labels

'plotting the result'
plt.scatter(trees_x['tree_id'],trees_x['tree_dbh'], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)
plt.xlabel("tree_id")
plt.ylabel("tree_dbh")
