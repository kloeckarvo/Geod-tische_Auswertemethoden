import numpy as np
from math import sqrt, atan2, pi, cos, sin

# Gegebene Werte für die Passpunkte
passpunkte = [
    ("A", 1.872, 30.902, 494086.578, 5795436.483),  # (Name, y, x, Y, X)
    ("B", 47.212, 22.842, 494043.439, 5795420.368),
    ("C", 24.132, 2.322, 494052.863, 5795449.779),
    ("D", 72.202, 5.762, 494013.233, 5795422.358),
    ("E", 35.742, 34.482, 494059.229, 5795416.173),
]

neupunkt = ("P", 50.017, 19.810)  # (Name, y, x) des zu transformierenden Punktes

# Anzahl der Passpunkte
n = len(passpunkte)
print(f"-- gegeben: {n} Passpunkte im yx-/Quell- und YX-/Zielsystem")
for i, (name, yy, xx, YY, XX) in enumerate(passpunkte, start=1):
    print(f"#{i:02d}: [ {name:10} (y={yy:+10.3f} m, x={xx:+10.3f} m), (Y={YY:+10.3f} m, X={XX:+10.3f} m) ]")
print()

# Schwerpunktberechnung für die Passpunkte im Quell- und Zielsystem
yS = np.mean([yy for _, yy, _, _, _ in passpunkte])
xS = np.mean([xx for _, _, xx, _, _ in passpunkte])
YS = np.mean([YY for _, _, _, YY, _ in passpunkte])
XS = np.mean([XX for _, _, _, _, XX in passpunkte])
print(f"Schwerpunktkoordinaten: yS={yS:+10.3f} m, xS={xS:+10.3f} m; YS={YS:+10.3f} m, XS={XS:+10.3f} m")
print()

# Erstellung der Matrizen A und b für die Normalgleichungen (Berücksichtigung der Helmert-Transformation)
A = np.zeros((2 * n, 4))
b = np.zeros((2 * n, 1))
for i, (name, yy, xx, YY, XX) in enumerate(passpunkte):
    # Schwerpunktreduzierte Koordinaten
    yy, xx, YY, XX = yy - yS, xx - xS, YY - YS, XX - XS
    A[2 * i] = [YY, -XX, 1, 0]
    A[2 * i + 1] = [XX, YY, 0, 1]
    b[2 * i] = [yy]
    b[2 * i + 1] = [xx]

print("-- Fehlergleichungen")
print(np.hstack((A, b)))
print()

# Berechnung der Normalgleichungen
N = A.T @ A
y = A.T @ b
print("-- Normalgleichungen")
print(np.hstack((N, y)))
print()

# Lösung des Gleichungssystems für die Transformationsparameter
Q = np.linalg.inv(N)
params = np.linalg.solve(N, y).flatten()
a, b, Ty, Tx = params  # a, b: Skalierungs- und Rotationsparameter; Ty, Tx: Translationen
print(f"Transformationsparameter:")
print(f"  a (cos) = {a:+.6f}")
print(f"  b (sin) = {b:+.6f}")
print(f"  Ty = {Ty:+.3f} m")
print(f"  Tx = {Tx:+.3f} m")

# Berechnung des Maßstabs und des Drehwinkels
s = sqrt(a**2 + b**2)  # Skalierung
sppm = (s - 1) * 1.0E6  # Skalierung in ppm
theta = atan2(b, a) * 180 / pi  # Drehwinkel in Grad
print(f"Skalierung       s={s:12.7f}, (s-1)={sppm:.1f} ppm (part per million)")
print(f"Drehwinkel     theta={theta:+10.3f} deg")
print()

# Transformation des zu transformierenden Punkts P
yP, xP = neupunkt[1] - yS, neupunkt[2] - xS
YP_trans = Tx + s * (cos(theta * pi / 180) * yP - sin(theta * pi / 180) * xP)
XP_trans = Ty + s * (sin(theta * pi / 180) * yP + cos(theta * pi / 180) * xP)
print(f"Transformierte Koordinaten von Punkt P: YP'={YP_trans:+15.3f} m, XP'={XP_trans:+15.3f} m")
print()

# Berechnung der absoluten Gauß-Krüger-Koordinaten für den transformierten Punkt
YP_gk = YS + YP_trans
XP_gk = XS + XP_trans

# Ausgabe in Gauß-Krüger-Koordinaten
print(f"Transformierte Koordinaten von Punkt P in Gauß-Krüger: YP_gk={YP_gk:+15.3f} m, XP_gk={XP_gk:+15.3f} m")
print()

# Berechnung der Restklaffungen
v = A @ np.array([[a], [b], [Ty], [Tx]]) - b
print("-- Restklaffungen (in mm)")
for i, passpunkt in enumerate(passpunkte):
    name = passpunkt[0]  # Nur den Namen extrahieren
    vy, vx = -v[2 * i, 0] * 1000, -v[2 * i + 1, 0] * 1000
    print(f"#{i+1:02d}: {name:10} vy={vy:+6.1f} mm, vx={vx:+6.1f} mm")

# Redundanzanteile (optional, wenn benötigt)
R = np.diag(A @ Q @ A.T)  # Redundanzmatrix Diagonale
print("\nRedundanzanteile:")
for i, r in enumerate(R):
    print(f"{i+1:02d}: {r:.4f}")
