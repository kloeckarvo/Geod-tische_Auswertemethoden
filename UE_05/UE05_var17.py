import numpy as np
from math import sqrt, atan2, pi


# Gegebene Werte für die Passpunkte
passpunkte = [
   ("A", 19.16, 15.09, 3494082.69, 5795440.82),  # (Name, y, x, r, h)
   ("B", 39.18, 37.04, 3494059.26, 5795422.54),  # (Name, y=Rechtswert(N), x=Hochwert(E), r=Rechtswert(N), h=Hochwert(E))
]
neupunkt = ("P", 18.05, 24.97)  # (Name, y=Rechtswert(N), x=Hochwert(E)) des zu transformierenden Punktes


# Anzahl der Passpunkte
n = len(passpunkte)
print(f"-- gegeben: {n} Passpunkte im yx-/Quell- und hr-/Zielsystem")
for i, (name, yy, xx, rr, hh) in enumerate(passpunkte, start=1):
   print(f"#{i:02d}: [ {name:10} (y={yy:+15.3f} m, x={xx:+15.3f} m), (r={rr:+15.3f} m, h={hh:+15.3f} m) ]")
print()


# Schwerpunktberechnung für die Passpunkte
yS = np.mean([yy for _, yy, _, _, _ in passpunkte])
xS = np.mean([xx for _, _, xx, _, _ in passpunkte])
rS = np.mean([rr for _, _, _, rr, _ in passpunkte])
hS = np.mean([hh for _, _, _, _, hh in passpunkte])
print(f"Schwerpunktkoordinaten: yS={yS:+10.3f} m, xS={xS:+10.3f} m; rS={rS:+10.3f} m, hS={hS:+10.3f} m")
print()


# Erstellung der Matrizen A und b für die Normalgleichungen
n, u = 2 * len(passpunkte), 4
f = n - u  # Freiheitsgrad
print(f"n={n} Messungen, u={u} Unbekannte, f={f} Freiheitsgrade")
print()


A = np.zeros((n, u))
b = np.zeros((n, 1))
for i, (name, yy, xx, rr, hh) in enumerate(passpunkte):
   # Reduzierte Koordinaten (zentriert)
   yy, xx, rr, hh = yy - yS, xx - xS, rr - rS, hh - hS
   A[2 * i] = [1, 0, yy, -xx]
   A[2 * i + 1] = [0, 1, xx, yy]
   b[2 * i] = [rr]
   b[2 * i + 1] = [hh]


print("-- Fehlergleichungen")
print(np.hstack((A, b)))
print()


# Berechnung der Normalgleichungen
N = A.T @ A
y = A.T @ b
print("-- Normalgleichungen")
print(np.hstack((N, y)))
print()


# Lösung des Gleichungssystems
Q = np.linalg.inv(N)
r0, h0, a, o = np.linalg.solve(N, y).flatten()
print(f"r0={r0:+.3f} m, h0={h0:+.3f} m, a={a:+.6f}, o={o:+.6f}")
print()


# Berechnung des Maßstabs und des Drehwinkels
q = sqrt(a**2 + o**2)
qppm = (q - 1) * 1.0E6
phideg = atan2(o, a) * 180.0 / pi
print(f"Maßstab       q={q:12.7f}, (q-1)={qppm:.1f} ppm (part per million), [mm/km]")
print()
print(f"Verdrehung     phi={phideg:+10.3f} deg")
print()


# Transformation des zu transformierenden Punktes P
yP, xP = neupunkt[1] - yS, neupunkt[2] - xS
rP_trans = r0 + (a * yP - o * xP)
hP_trans = h0 + (o * yP + a * xP)
print(f"Transformierte Koordinaten von Punkt P: rP'={rP_trans:+15.3f} m, hP'={hP_trans:+15.3f} m")
print()


# Berechnung der absoluten Gauß-Krüger-Koordinaten
rP_gk = rS + rP_trans
hP_gk = hS + hP_trans


# Ausgabe in Gauß-Krüger-Koordinaten
print(f"Transformierte Koordinaten von Punkt P in Gauß-Krüger: rP_gk={rP_gk:+15.3f} m, hP_gk={hP_gk:+15.3f} m")
print()


# # Berechnung der Restklaffungen
# v = A @ np.array([[r0], [h0], [a], [o]]) - b
# print("-- Restklaffungen (in mm)")
# for i, passpunkt in enumerate(passpunkte):
#     name = passpunkt[0]  # Nur den Namen extrahieren
#     kr, kh = -v[2 * i, 0] * 1000, -v[2 * i + 1, 0] * 1000
#     print(f"#{i+1:02d}: {name:10} kr={kr:+6.1f} mm, kh={kh:+6.1f} mm")
#
# # Redundanzanteile (optional, wenn benötigt)
# R = np.diag(A @ Q @ A.T)  # Redundanzmatrix Diagonale
# print("\nRedundanzanteile:")
# for i, r in enumerate(R):
#     print(f"{i+1:02d}: {r:.4f}")
# print()
# # Erzeugung der Diagonalmatrix mit denselben Einträgen
# R_matrix = np.diag(R)
#
# # Ausgabe der Diagonalmatrix
# print("Redundanzmatrix als Diagonalmatrix:")
# print(R_matrix)

