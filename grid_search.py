import itertools
import multiprocess
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


# Parámetros para Random Forest
param_grid_rf = {
    'n_estimators': [10, 20, 40, 100],
    'criterion': ['gini', 'entropy', 'log_loss']
}

# Generar combinaciones para Random Forest
keys_rf, values_rf = zip(*param_grid_rf.items())
combinations_rf = [dict(zip(keys_rf, v)) for v in itertools.product(*values_rf)]


# Función a paralelizar
def evaluate_set(hyperparameter_set, lock, X_train, y_train, X_test, y_test):
    """
    Evaluate a set of hyperparameters
    Args:
    hyperparameter_set: a list with the set of hyperparameters to be evaluated
    """
    for s in hyperparameter_set:
        clf = RandomForestClassifier()
        clf.set_params(n_estimators=s['n_estimators'], criterion=s['criterion'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Exclusión mutua
        lock.acquire()
        print(f'Accuracy en el proceso con params {s}:', accuracy_score(y_test, y_pred))
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


if __name__ == '__main__':
    # Ruta del archivo
    ruta = "Brain_Tumor.csv"

    # Preprocesar el dataset
    dataset_balanceado = preprocesar_dataset(ruta)

    # Separar características y etiquetas
    X = dataset_balanceado.drop(columns=['Class']).to_numpy()  # Convertir a numpy.ndarray
    y = dataset_balanceado['Class'].to_numpy()  # Convertir a numpy.ndarray

    # Separar en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    # Ahora se evaluará con más procesos
    threads = []
    N_THREADS = 7
    splits = nivelacion_cargas(combinations_rf, N_THREADS)
    lock = multiprocess.Lock()

    for i in range(N_THREADS):
        # Se generan los procesos de procesamiento
        threads.append(
            multiprocess.Process(target=evaluate_set, args=(splits[i], lock, X_train, y_train, X_test, y_test)))

    start_time = time.perf_counter()

    # Se lanzan a ejecución
    for thread in threads:
        thread.start()

    # y se espera a que todos terminen
    for thread in threads:
        thread.join()

    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")

