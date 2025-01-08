import numpy as np

np.set_printoptions(suppress=True)

# Gegebene Daten aus der Tabelle (inklusive Standardabweichungen sX, sY)
pass_pkt = np.array([
    [0.312, 29.342, 494087.123, 5795438.615, 0.010, 0.010],
    [45.652, 21.282, 494043.984, 5795422.500, 0.020, 0.020],
    [22.572, 0.762, 494053.418, 5795451.911, 0.010, 0.010]
])

neu_pkt = np.array([
    [50.017, 19.810]
])

# Designmatrix, b-Vektor und Gewichtsmatrix initialisieren
A = []
b = []
weights = []

# Generiere die Designmatrix, den b-Vektor und die Gewichtsmatrix
for y, x, Y, X, sY, sX in pass_pkt:
    A.append([x, y, 1, 0, 0, 0, -x*X, -y*X])
    A.append([0, 0, 0, x, y, 1, -x*Y, -y*Y])
    b.append(X)
    b.append(Y)
    weights.append(1 / sX**2)  # Gewicht für X
    weights.append(1 / sY**2)  # Gewicht für Y

A = np.array(A)
b = np.array(b)
P = np.diag(weights)  # Diagonale Gewichtsmatrix

# Gewichtete Normalgleichungen
N = A.T @ P @ A  # Normalgleichungsmatrix
w = A.T @ P @ b  # Rechter Hand-Vektor

print()
print(A)
print()
print(b)
print()
print(P)
print()

# Lösung des gewichteten Gleichungssystems
parameters = np.linalg.solve(N, w)

# Parameter anzeigen
print("Transformationsparameter:")
print(f"a_1 = {parameters[0]:.7f}, b_1 = {parameters[3]:.7f}")
print(f"a_2 = {parameters[1]:.7f}, b_2 = {parameters[4]:.7f}")
print(f"a_3 = {parameters[2]:.7f}, b_3 = {parameters[5]:.7f}")
print(f"c_1 = {parameters[6]:.7f}, c_2 = {parameters[7]:.7f}")

# Transformierte Punkte berechnen
T = []
for y, x in neu_pkt:
    X_new = (x * parameters[0] + y * parameters[1] + parameters[2]) / (1 + parameters[6] * x + parameters[7] * y)
    Y_new = (x * parameters[3] + y * parameters[4] + parameters[5]) / (1 + parameters[6] * x + parameters[7] * y)
    T.append([Y_new, X_new])

# Ergebnisse anzeigen
print("\nTransformierte Punkte:")
print("    Y_transformed    X_transformed")
for idx, (Y, X) in enumerate(T, start=11):
    print(f"{idx:<10} {Y:<15.3f} {X:<15.3f}")

