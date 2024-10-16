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

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name = "birch-rg2.arff"

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

# Création d'une boucle pour récupérer la valeur de l'inertie pour chaque clustering
listeInertie = []
nbre_clusterMax = 20
for i in range(1, nbre_clusterMax):
    model = cluster.KMeans(n_clusters=i, init='k-means++', n_init=10)
    model.fit(datanp)
    inertie = model.inertia_
    listeInertie.append(inertie)

# Affichage de l'inertie en fonction du nombre de clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, nbre_clusterMax), listeInertie, marker='o')
plt.title("Inertie en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.xticks(range(1, nbre_clusterMax))
plt.grid()
plt.show()

# Calcul des dérivées
#d_1 = np.diff(listeInertie)  # Dérivée première
#d_2 = np.diff(d_1)  # Dérivée seconde

# Identification du "point d'inflexion" de la courbe
#nombre_clusters_ideal = nombre_clusters_ideal = 


#print(f"Le nombre de clusters idéal (point d'inflexion) est : {nombre_clusters_ideal}")


tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
#plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

#print("nb clusters =",nbre_clusterMax,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

#from sklearn.metrics.pairwise import euclidean_distances
#dists = euclidean_distances(centroids)
#print(dists)

