import numpy as np
import sympy as sp
from sympy import Matrix

# Designmatrix A
A = Matrix([
    [202.17, 1.],
    [204.04, 1.],
    [209.98, 1.],
    [213.52, 1.],
    [214.89, 1.],
    [217.01, 1.],
    [220.33, 1.],
    [225.06, 1.],
    [227.88, 1.],
    [233.01, 1.],
    [234.22, 1.]
])

# b-Vektor (Messwerte)
b = Matrix([
    [102.713],
    [102.712],
    [102.707],
    [102.710],
    [102.711],
    [102.702],
    [102.702],
    [102.700],
    [102.701],
    [102.699],
    [102.703]
])

# Genauigkeiten (Standardabweichungen)
genauigkeiten = Matrix([
    [0.005],
    [0.003],
    [0.003],
    [0.005],
    [0.005],
    [0.003],
    [0.003],
    [0.003],
    [0.003],
    [0.004],
    [0.005]
])

# Berechnung der Varianzen (quadrierte Standardabweichungen)
varianzen = genauigkeiten.applyfunc(lambda x: x**2)

# Berechnung der Kehrwerte der Varianzen (Gewichte)
gewichte = varianzen.applyfunc(lambda x: 1/x)

# Gewichtsmatrix P
P = sp.diag(*gewichte)

# Berechnung der transponierten Matrix A
A_T = A.T

# Normalgleichungsmatrix mit und ohne Gewichte
N = A_T * A
N_P = A_T * P * A

# Kofaktormatrix Q aus der invertierten Matrix N_P
Q = N_P.inv()

# Berechnung der Koeffizienten (Steigung m und Achsenabschnitt n) für die Geradengleichung
ATPA = A_T * P * A
ATPA_inv = ATPA.inv()
x = ATPA_inv * (A_T * P * b)

# Parameter der Geraden (x[0] = m, x[1] = n)
m = x[0]
n = x[1]

# Vektor der geschätzten Modellwerte (l)
l = A * x

# Vektor der Verbesserungen (v)
v = b - l

# Funktion zur Berechnung der Geradengleichung für einen gegebenen x-Wert
def lineare_funktion(x_wert, m, n):
    return m * x_wert + n

# Beispiel x-Wert zur Berechnung
x_wert = 234.22
y_wert = lineare_funktion(x_wert, m, n)

# Ausgabe des Ergebnisses
print(f"Für den x-Ausgangswert = {x_wert:.3f} beträgt der (verbesserte) y-Wert: {y_wert:.3f}")
print(f"Funktionsgleichung: f(x) = {m:.5f}x + {n:.5f}")

# Debug-Ausgaben
print("Designmatrix A:")
print(A)
print("Transponierte Matrix A_T:")
print(A_T)
print("b-Vektor:")
print(b)
print("Gewichtsmatrix P:")
print(P)
print("Normalgleichungsmatrix N:")
print(N)
print("Normalgleichungsmatrix mit Gewichten N_P:")
print(N_P)
print("Kofaktormatrix Q:")
print(Q)
print("Koeffizienten x:")
print(x)
print("Geschätzte Modellwerte l:")
print(l)
print("Verbesserungen v:")
print(v)