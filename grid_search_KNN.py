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

# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, X_train, y_train, X_test, y_test, results):
    for s in hyperparameter_set:
        clf = KNeighborsClassifier()
        clf.set_params(n_neighbors=s['n_neighbors'],
                       algorithm=s['algorithm'],
                       metric=s['metric'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        lock.acquire()
        print(f'Accuracy en el proceso con params {s}:', accuracy_score(y_test, y_pred))
        results.append((s, accuracy))  # Guardar la combinacion de hiperparametros y el accuracy
        lock.release()


# Función para preprocesar el dataset
def preprocesar_dataset(archivo):
    dataset = pd.read_csv(archivo)
    dataset = dataset.drop(columns=["Image"])  # eliminar la primera columna

    conteo_clases = dataset['Class'].value_counts()  # contar el numero de instancias de cada clase

    # imprimir el conteo de clases inicial
    print("Conteo de clases:")
    print(conteo_clases)

    clase_min = conteo_clases.min()  # Se obtiene el minimo de instancias por clase para balancear el dataset
    nuevo_dataset = []

    # Iterar sobre cada clase y tomar una muestra de tamaño clase_min
    for clase in dataset['Class'].unique():
        subconjunto = dataset[dataset['Class'] == clase].sample(clase_min, random_state=42)
        nuevo_dataset.append(subconjunto)

    # Combinar los subconjuntos balanceados en un solo dataframe
    dataset_balanced = pd.concat(nuevo_dataset).reset_index(drop=True)
    dataset_balanced = dataset_balanced.sort_values(by='Class').reset_index(drop=True)

    # imprimir el conteo de instancias por clase actual
    print("\nConteo de clases (despues de preprocesar):")
    print(dataset_balanced['Class'].value_counts())

    return dataset_balanced