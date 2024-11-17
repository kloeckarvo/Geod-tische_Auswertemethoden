import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gegebene Passpunkte und Neupunkte
passpunkte = pd.DataFrame({
    'Nr': [1, 2, 3, 4],
    'y': [4.247, 42.723, 89.835, 68.523],
    'x': [15.005, 66.376, 57.900, 2.136],
    'Y': [10.000, 10.000, 60.000, 60.000],
    'X': [10.000, 80.000, 80.000, 10.000]
})

neupunkte = pd.DataFrame({
    'Nr': [11, 12, 13, 14],
    'y': [33.688, 46.624, 66.482, 56.052],
    'x': [37.464, 56.662, 53.011, 33.187]
})


def build_matrices(passpunkte):
    """
    Erstellt Matrizen A und B für die projektive Transformation.
    """
    N = len(passpunkte)
    A = np.zeros([2 * N, 8])
    B = np.zeros([2 * N, 1])

    for i in range(N):
        y, x, Y, X = passpunkte.iloc[i][['y', 'x', 'Y', 'X']]
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * X, -y * X]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * Y, -y * Y]
        B[2 * i] = X
        B[2 * i + 1] = Y

    return A, B


def compute_transformation(A, B):
    """
    Berechnet die Transformationsparameter.
    """
    N = A.T @ A
    Q = np.linalg.inv(N)
    HW = Q @ A.T
    X = HW @ B
    return X.flatten()


def transform_neupunkte(neupunkte, X):
    """
    Transformiert neue Punkte mit den berechneten Parametern.
    """
    result = []
    for _, row in neupunkte.iterrows():
        Nr, y, x = row[['Nr', 'y', 'x']]
        denom = X[6] * x + X[7] * y + 1
        X_trans = (X[0] * x + X[1] * y + X[2]) / denom
        Y_trans = (X[3] * x + X[4] * y + X[5]) / denom
        result.append([Nr, y, x, round(Y_trans, 3), round(X_trans, 3)])
    return pd.DataFrame(result, columns=['Nr', 'y', 'x', "Y'", "X'"])


def visualize_results(passpunkte, neupunkte, neupunkte_trans):
    """
    Visualisiert Passpunkte und transformierte Punkte.
    """
    plt.figure(figsize=(10, 6))
    # Passpunkte (Original und Zielsystem)
    plt.scatter(passpunkte['x'], passpunkte['y'], label='Passpunkte (Quellsystem)', c='purple', marker='o')
    plt.scatter(passpunkte['X'], passpunkte['Y'], label='Passpunkte (Zielsystem)', c='cyan', marker='s')
    # Neue Punkte (Original und transformiert)
    plt.scatter(neupunkte['x'], neupunkte['y'], label='Neue Punkte (Original)', c='magenta', marker='^')
    plt.scatter(neupunkte_trans["X'"], neupunkte_trans["Y'"], label='Neue Punkte (Transformiert)', c='black', marker='x')

    plt.legend()
    plt.title('Projektive Transformation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()


# Hauptprogramm

# Matrizen erstellen
A, B = build_matrices(passpunkte)

# Transformation berechnen
params = compute_transformation(A, B)

# Neue Punkte transformieren
neupunkte_trans = transform_neupunkte(neupunkte, params)

# Ergebnisse anzeigen
print("\nTransformierte Koordinaten:")
print(neupunkte_trans)

# Visualisierung
visualize_results(passpunkte, neupunkte, neupunkte_trans)
