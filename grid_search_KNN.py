import itertools
import multiprocess
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Metodo para hacer nivelación de cargas
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Parametros de KNN
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['euclidean', 'manhattan']
}

# Generar combinaciones para KNN
keys_knn, values_knn = zip(*param_grid_knn.items())
combinations_knn = [dict(zip(keys_knn, v)) for v in itertools.product(*values_knn)]
#print(len(combinations_knn))