import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


###################################################################
# Exemple : A CHANGER


path = './artificial/'
name="donut1.arff"

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



### Déterminer epsilon :
# extrait du sujet de TP (page 6)
# Distances aux k plus proches voisins
# Donnees dans X
k=5
neigh = NearestNeighbors(n_neighbors =k)
neigh.fit(datanp)
distances,indices = neigh.kneighbors (datanp)
# distance moyenne sur les k plus proches voisins
# en retirant le point " origine "
newDistances = np. asarray ([np. average ( distances [i][1:]) for i in range (0,
distances . shape [0])])
# trier par ordre croissant
distancetrie = np. sort ( newDistances )
plt. title (" Plus proches voisins "+str(k))
plt. plot ( distancetrie )
plt. show ()

#######################################################################