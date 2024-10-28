import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


###################################################################


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

### Déterminer min_samples :
# Calcul du coefficient silhouette, de l'indice de Davies Bouldin (DB) pour chaque valeur de min_samples
listeSilhouette = []
listeDB=[]
listeCalin=[]
minSamplesMax = 20

for i in range(2, minSamplesMax + 1):  # Commencer à 2 car silhouette_score nécessite au moins 2 clusters
    model = cluster.DBSCAN(eps=0.005, min_samples=i) #valeurs d'epsilon déterminé à l'aide de la courbe issue de "DBScan_evalEps.py"
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

# Affichage du score silhouette en fonction de la valeur de min_samples
plt.figure(figsize=(10, 6))
plt.plot(range(2, minSamplesMax + 1), listeSilhouette, marker='o', color='blue')
plt.title("Score de silhouette en fonction de la valeur de min_samples")
plt.xlabel("Valeur de min_samples")
plt.ylabel("Score de silhouette moyen")
plt.xticks(range(2, minSamplesMax + 1))
plt.grid()
plt.show()

# Affichage de l'indice de DB en fonction de la valeur de min_samples 
plt.figure(figsize=(10, 6))
plt.plot(range(2, minSamplesMax+1), listeDB, marker='o', color='green')
plt.title("Coefficient de DB en fonction de la valeur de min_samples")
plt.xlabel("Valeur de min_samples")
plt.ylabel("Coefficient de DB")
plt.xticks(range(2, minSamplesMax+1))
plt.grid()
plt.show() 

# Affichage de l'indice de Calinski en fonction de la valeur de min_samples 
plt.figure(figsize=(10, 6))
plt.plot(range(2, minSamplesMax+1), listeCalin, marker='o', color='green')
plt.title("Coefficient de Calinski-Harabasz en fonction de la valeur de min_samples")
plt.xlabel("Valeur de min_samples")
plt.ylabel("Coefficient de Calinski-Harabasz")
plt.xticks(range(2, minSamplesMax+1))
plt.grid()
plt.show()
