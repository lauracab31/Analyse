import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="complex9.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])


# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()



### FIXER la distance
# 
tps1 = time.time()
seuil_dist=200
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=4
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='average', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

# Calcul du coefficient silhouette, de l'indice de Davies Bouldin (DB) pour chaque nombre de clusters
listeSilhouette = []
listeDB=[]
listeCalin=[]
nbre_clusterMax = 20

for i in range(2, nbre_clusterMax + 1):  # Commencer à 2 car silhouette_score nécessite au moins 2 clusters
    model = cluster.AgglomerativeClustering(linkage='average', n_clusters=i)
    model.fit(datanp)

    # Calcul du score silhouette
    silhouette_avg = silhouette_score(datanp, model.labels_)
    listeSilhouette.append(silhouette_avg)

    #calcul de l'indice de DB
    db_score=davies_bouldin_score(datanp, model.labels_)
    listeDB.append(db_score)

    #calcul de l'indice de Calinski
    db_score=calinski_harabasz_score(datanp, model.labels_)
    listeCalin.append(db_score)

# Affichage du score silhouette en fonction du nombre de clusters
plt.figure(figsize=(10, 6))
plt.plot(range(2, nbre_clusterMax + 1), listeSilhouette, marker='o', color='blue')
plt.title("Score de silhouette en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Score de silhouette moyen")
plt.xticks(range(2, nbre_clusterMax + 1))
plt.grid()
plt.show()

# Affichage de l'indice de DB en fonction du nombre de clusters 
plt.figure(figsize=(10, 6))
plt.plot(range(2, nbre_clusterMax+1), listeDB, marker='o', color='green')
plt.title("Coefficient de DB en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Coefficient de DB")
plt.xticks(range(2, nbre_clusterMax+1))
plt.grid()
plt.show()

# Affichage de l'indice de Calinski en fonction du nombre de clusters 
plt.figure(figsize=(10, 6))
plt.plot(range(2, nbre_clusterMax+1), listeCalin, marker='o', color='green')
plt.title("Coefficient de Calinski-Harabasz en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters")
plt.ylabel("Coefficient de Calinski-Harabasz")
plt.xticks(range(2, nbre_clusterMax+1))
plt.grid()
plt.show()

#######################################################################