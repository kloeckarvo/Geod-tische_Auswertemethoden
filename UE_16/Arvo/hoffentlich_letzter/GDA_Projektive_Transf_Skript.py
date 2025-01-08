import numpy as np
import pandas as pd
import sympy as sp
import math

np.set_printoptions(suppress=True)

# Gegebene Daten aus der Tabelle
pass_pkt = np.array([
    [4.247, 15.055, 10.000, 10.000],
    [42.723, 66.376, 10.000, 80.000],
    [89.885, 57.850, 60.000, 80.000],
    [68.523, 2.136, 60.000, 10.000]
])

neu_pkt = np.array([
    [33.688, 37.464],
    [46.624, 56.662],
    [66.482, 53.011],
    [56.052, 33.187]
])

# Designmatrix und b-Vektor initialisieren
A = []
b = []

# Generiere die Designmatrix und den b-Vektor
for y, x, Y, X in pass_pkt:
    A.append([x, y, 1, 0, 0, 0, -x*X, -y*X])
    A.append([0, 0, 0, x, y, 1, -x*Y, -y*Y])
    b.append(X)
    b.append(Y)

A = np.array(A)
b = np.array(b)

print()
print("Designmatrix A:")
print(A)
print()
print("b-Vektor:")
print(b)
print()

# Lösung des Gleichungssystems
parameters = np.linalg.lstsq(A, b, rcond=None)[0]  # Lösungsvektor

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

