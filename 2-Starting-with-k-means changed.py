"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples


##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name = "impossible.arff"

databrut = arff.loadarff(open(path + str(name), 'r'))
datanp = np.array([[x[0], x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
print("---------------------------------------")
print("Affichage données initiales            " + str(name))
f0 = datanp[:, 0]  # tous les élements de la première colonne
f1 = datanp[:, 1]  # tous les éléments de la deuxième colonne

plt.scatter(f0, f1, s=8)
plt.title("Données initiales : " + str(name))
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()

# Calcul de l'inertie et du score silhouette pour chaque nombre de clusters
listeInertie = []
listeSilhouette = []
nbre_clusterMax = 20

for i in range(2, nbre_clusterMax + 1):  # Commencer à 2 car silhouette_score nécessite au moins 2 clusters
    model = cluster.KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
    model.fit(datanp)
    
    # Calcul de l'inertie
    inertie = model.inertia_
    listeInertie.append(inertie)
    
    # Calcul du score silhouette
    silhouette_avg = silhouette_score(datanp, model.labels_)
    listeSilhouette.append(silhouette_avg)

# Affichage du score silhouette en fonction du nombre de clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, nbre_clusterMax + 1), listeSilhouette, marker='o', color='blue')
plt.title("Score de silhouette en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score de silhouette moyen")
plt.xticks(range(2, nbre_clusterMax + 1))
plt.grid()
plt.show()

# Affichage de l'inertie en fonction du nombre de clusters (optionnel)
plt.figure(figsize=(10, 6))
plt.plot(range(2, nbre_clusterMax + 1), listeInertie, marker='o', color='green')
plt.title("Inertie en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.xticks(range(2, nbre_clusterMax + 1))
plt.grid()
plt.show()

#print("nb clusters =",nbre_clusterMax,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

#from sklearn.metrics.pairwise import euclidean_distances
#dists = euclidean_distances(centroids)
#print(dists)