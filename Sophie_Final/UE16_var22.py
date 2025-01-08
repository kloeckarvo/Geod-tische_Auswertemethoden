import numpy as np

# Eingabedaten
passpunkte = np.array([
    [0.624, 29.654, 494087.014, 5795438.189, 0.010, 0.020],
    [45.964, 21.594, 494043.875, 5795422.074, 0.020, 0.020],
    [22.884, 1.074, 494053.307, 5795451.485, 0.010, 0.010]
])

neue_punkte = np.array([
    [50.017, 19.810]
])

# Initialisierung der Matrizen
A = []
b = []
gewichte = []

# Aufbau der Designmatrix, des b-Vektors und der Gewichtsmatrix
for y, x, Y, X, sY, sX in passpunkte:
    A.append([x, y, 1, 0, 0, 0, -x*X, -y*X])
    A.append([0, 0, 0, x, y, 1, -x*Y, -y*Y])
    b.append(X)
    b.append(Y)
    gewichte.append(1 / sX**2)
    gewichte.append(1 / sY**2)

A = np.array(A)
b = np.array(b)
P = np.diag(gewichte)

# Normalisierung der Gewichtsmatrix
P /= np.max(P)

# Gewichtete Normalgleichungen
N = A.T @ P @ A
w = A.T @ P @ b

# Lösung des Gleichungssystems
parameter = np.linalg.solve(N, w)

# Ausgabe der Transformationsparameter
print("Transformationsparameter:")
print(f"a1 = {parameter[0]:.7f}, b1 = {parameter[3]:.7f}")
print(f"a2 = {parameter[1]:.7f}, b2 = {parameter[4]:.7f}")
print(f"a3 = {parameter[2]:.7f}, b3 = {parameter[5]:.7f}")
print(f"c1 = {parameter[6]:.7f}, c2 = {parameter[7]:.7f}")

# Berechnung der transformierten Punkte
transformierte_punkte = []
for y, x in neue_punkte:
    X_neu = (x * parameter[0] + y * parameter[1] + parameter[2]) / (1 + parameter[6] * x + parameter[7] * y)
    Y_neu = (x * parameter[3] + y * parameter[4] + parameter[5]) / (1 + parameter[6] * x + parameter[7] * y)
    transformierte_punkte.append([Y_neu, X_neu])

# Ausgabe der transformierten Punkte
np.set_printoptions(suppress=True)
print("\nTransformierte Punkte:")
print("    Y_transformed    X_transformed")
for idx, (Y, X) in enumerate(transformierte_punkte, start=11):
    print(f"{idx:<10} {Y:<15.3f} {X:<15.3f}")

# Ausgabe der Matrizen
print("\nDesignmatrix A:")
print(A)
print("\nb-Vektor:")
print(b)
print("\nGewichtsmatrix P:")
print(P)