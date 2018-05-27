# coding: utf-8

# # Dependencies

import math
import numpy as np
import pandas as pd
import os
import datetime
from lapjv import lapjv
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# # Functions

def cart2pol(v):
    rho = np.sqrt(v[0]**2 + v[1]**2)
    phi = np.arctan2(v[1], v[0])
    return(rho, phi)


def pol2cart(p):
    x = p[0] * np.cos(p[1])
    y = p[0] * np.sin(p[1])
    return(x, y)


def kmeans_balanced(locations, k=10, numiter=10):
    #
    # Balanced k-means clustering via linear assignment problem (LAP)
    #
    # 0) define algorithm parameters
    # TODO: smart cluster initialization for faster convergence.
    # locations = df.as_matrix()
    N = locations.shape[0]
    n_k = round(N / k) + 1
    idx = np.random.choice(N, k, replace=False)
    C = locations[idx, :]

    # I.a) Assignment problem
    for i in range(numiter):
        C_rep = np.tile(C, (n_k, 1))
        C_rep = C_rep[0:N, :]
        cost_matrix = cdist(locations, C_rep)
        assignment = lapjv(cost_matrix)

        # I.b) Points -> clusters
        cluster_idx = np.tile(np.arange(0, k)[np.newaxis].transpose(), (n_k, 1))
        cluster_idx = cluster_idx[0:N, :]

        assignment_inv = np.argsort(assignment[1])
        # II.a) Compute centroids
        temp = pd.DataFrame(locations, columns=['x', 'y'])
        temp['cluster'] = cluster_idx[assignment_inv]

        C = temp.groupby('cluster').mean()
    return(temp)

# #  Clusterization


num_clusters = 17


depot = np.array([-35.4003338, 149.1557231])


locations_df = pd.read_csv('locations.csv')

locations = locations_df.as_matrix()


# # Balanced K-Means
l = locations
# 1) Shift the data
l = l - depot

# 2) perform conversion to polar coordinates

pol = np.apply_along_axis(cart2pol, 1, l)

# 3) perform the log transform of the data
log_pol = pol
log_pol[:, 0] = np.log(log_pol[:, 0])

# 4) Normalize the data and define control parameters
beta = 0.3
log_pol[:, 0] = beta * log_pol[:, 0]


t0 = datetime.datetime.now()
result = kmeans_balanced(log_pol, k=num_clusters)
t1 = datetime.datetime.now()
y_pred = result.cluster.as_matrix()
print(datetime.datetime.now() - t0)
locations_df['group'] = y_pred


for group in locations_df['group'].unique():
    print(locations_df[locations_df['group'] == group].shape[0])

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
clusters_color = [colors[int(c)] for c in y_pred]

plt.figure(figsize=(20, 15))
plt.subplot(2, 1, 1)
plt.title('Balanced K-Means -  Clusters: {1}'.format(len(locations), len(set(clusters_color))), size=18)
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plt.plot(locations[:, 1], locations[:, 0], 'o', markerfacecolor='cyan',
         markeredgecolor='blue', markersize=3)
plt.plot(depot[1], depot[0], '*', markerfacecolor='yellow',
         markeredgecolor='k', markersize=25)
plt.subplot(2, 1, 2)
plt.scatter(locations[:, 1], locations[:, 0], color=clusters_color, s=30)
plt.plot(depot[1], depot[0], '*', markerfacecolor='yellow',
         markeredgecolor='k', markersize=25)
plt.show()
