import numpy as np
import sympy as sp
from sympy import *
from IPython.display import display, Math

# Designmatrix A (hier: Beizahl y-Achsenabschnitt jeweils = 1, Beobachtungen / Meßwerte = x-Werte)
A = Matrix([
    [121.405, 1.],
    [121.404, 1.],
    [121.403, 1.],
    [121.409, 1.],
    [121.411, 1.],
    [121.403, 1.],
    [121.406, 1.],
    [121.407, 1.],
    [121.410, 1.],
    [121.412, 1.],
    [121.416, 1.],
    [121.412, 1.]
])

# b-Vektor (hier: Zeitangaben = y-Werte)
b = Matrix([
    [202.17],
    [204.04],
    [209.98],
    [213.52],
    [214.89],
    [217.01],
    [220.33],
    [225.06],
    [227.88],
    [233.01],
    [234.22],
    [240.09]
])

# Option 1: Matrix A transponieren, Darstellung in Ausgangsform mit Exponent "T" zur Kennzeichnung
A_T = Transpose(A)

# Option 2: Matrix A transponieren, Darstellung mit vertauschten Spalten und Zeilen
A.T

# Genauigkeiten = Standardabweichungen für jeden einzelnen Meßwert (sympy-Matrix). Werte stammen aus dem Meßgerät und sind dort in der Regel hinterlegt, bzw. werden dort generiert.
genauigkeiten_sp = Matrix([
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
    [0.005],
    [0.003]
])

# 1.) gemessene Höhen außerhalb Design-Matrix (sympy-Matrix)
hoehen = Matrix([
    [121.405],
    [121.404],
    [121.403],
    [121.409],
    [121.411],
    [121.403],
    [121.406],
    [121.407],
    [121.410],
    [121.412],
    [121.416],
    [121.412]
])

# 2.) gemessene Höhen außerhalb Design-Matrix (numpy-Array)
hoehen_2 = np.array([
    [121.405,
     121.404,
     121.403,
     121.409,
     121.411,
     121.403,
     121.406,
     121.407,
     121.410,
     121.412,
     121.416,
     121.412]
])

# Generieren der Gewichtsmatrix P aus den "mitgelieferten" Genauigkeiten (= Standardabweichungen)
# Gegebene Genauigkeiten (Standardabweichungen)
genauigkeiten = sp.Matrix([
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
    [0.005],
    [0.003]
])

# Berechnung der Varianzen (Standardabweichungen quadrieren)
varianzen = genauigkeiten.applyfunc(lambda x: x**2)

# Berechnung der Kehrwerte der Varianzen (Gewichte)
gewichte = varianzen.applyfunc(lambda x: 1/x)

# Extrahieren der Diagonalelemente in eine Liste
diagonale_gewichte = [float(g[0]) for g in gewichte.tolist()]

# Berechnung des Minimums und Maximums der Diagonalelemente (Gewichte)
P_temp_min = min(diagonale_gewichte)
P_temp_max = max(diagonale_gewichte)

# Normalisierung der Diagonalelemente: (P - Min) / (Max - Min)
diagonale_gewichte_norm = [(g - P_temp_min) / (P_temp_max - P_temp_min) for g in diagonale_gewichte]

# Erstellung der Diagonalmatrix mit den normalisierten Diagonalelementen
P = sp.diag(*diagonale_gewichte_norm)

# Normalgleichungsmatrix OHNE Gewichte
N = A_T * A

# Normalgleichungsmatrix MIT Gewichten
N_P = A_T * P * A

# Kofaktormatrix Q aus der invertierten Matrix N_P ( = (A_T * P * A).inv() )
Q = N_P.inv()

#######################################################################################################################################################
#                                                                                                                                                     #
# Wir haben nun: Die Designmatrix A, die transponierte Designmatrix A_T, den Vektor b, die Gewichtsmatrix P, die Normalgleichungsmatrix N (bzw. N_P), #
# die Kofaktormatrix Q, den Vektor der geschätzten Parameter x, den Vektor der Modellwerte l und den Vektor der Verbesserungen v.                     #
# Zur Kontrolle: print(A) -> Run, print(A_T) -> Run, print(b) -> Run, print(P) -> Run, print(N) -> Run, print(N_P) -> Run, print(Q) -> Run,           #
# print(x) -> Run, print(l) -> Run, print(v) -> Run                                              #                                                    #                                       #
#                                                                                                                                                     #
#######################################################################################################################################################
#
# print(A)
# print(A_T)
# print(b)
# print(P)
# print(N)
# print(N_P)
# print(Q)
# print(x)
# print(l)
# print(v)
#
# in jupyter Notebook: print() kann bei Nutzung von sympy weggelassen werden, Bsp.: A, Shift + Enter oder "Run", Ausgabe der Matrix erfolgt unmittelbar!

# Ermitteln der Parameter (Steigung, y-Achsenabschnitt) für die Geradengleichung
ATPA = (A_T * P * A)
ATPA_inv = ATPA.inv()
x = ATPA_inv * (A_T * P * b)

# Vektor der geschätzten Modellwerte (l)
l = A * x

# Vektor der Verbesserungen (v)
v = b - l

# Probe: Geradengleichung als Funktion definieren, bekannten x-Wert wählen: y-Wert passend? (Achtung: Verbesserung berücksichtigt,
# deshalb ist hier nicht mehr der y-Ausgangswert aus der Tabelle zu erwarten!)
def lineare_funktion(x, m, n):
    return m * x + n

# Steigung (m) und y-Achsenabschnitt (n)
m = 2972.81245517731  # Steigung
n = -360698.096679688  # y-Achsenabschnitt

# Beispiel für einen gegebenen x-Wert
x_wert = 121.412

# Berechnung des y-Wertes
y_wert = lineare_funktion(x_wert, m, n)

# Ausgabe des Ergebnisses
print(f"Für den x-Ausgangswert = {x_wert:.3f} beträgt der (verbesserte) y-Wert: {y_wert:.3f}")

# Funktionsgleichung: f(x) = 2972.81245x - 360698.096679688