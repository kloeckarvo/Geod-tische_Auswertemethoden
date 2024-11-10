# Abgabe Übung 05
> **Arvo Klöck | 909003**  
> **10.11.2024**
    
```python
import numpy as np
from math import sqrt, atan2, pi


# Gegebene Werte für die Passpunkte
passpunkte = [
   ("A", 15,22, 19.03, 3494082.82, 5795440.63),  # (Name, y, x, r, h)
   ("B", 37.18, 39.05, 3494059.39, 5795422.35),  # (Name, y=Rechtswert(N), x=Hochwert(E), r=Rechtswert(N), h=Hochwert(E))
]
neupunkt = ("P", 20,00, 26,53)  # (Name, y=Rechtswert(N), x=Hochwert(E)) des zu transformierenden Punktes


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
```
# Gegeben: 2 Passpunkte im yx-/Quell- und hr-/Zielsystem

$
\begin{array}{ll}
\#01: & \left[ A \quad (y=+15.220 \, \text{m}, \, x=+19.030 \, \text{m}), \, (r=+3494082.820 \, \text{m}, \, h=+5795440.630 \, \text{m}) \right] \\
\#02: & \left[ B \quad (y=+37.180 \, \text{m}, \, x=+39.050 \, \text{m}), \, (r=+3494059.390 \, \text{m}, \, h=+5795422.350 \, \text{m}) \right]
\end{array}
$

## Schwerpunktkoordinaten
$y_S = +26.200 \, \text{m}, \quad x_S = +29.040 \, \text{m}; \quad r_S = +3494071.105 \, \text{m}, \quad h_S = +5795431.490 \, \text{m}$
## Messungen, Unbekannte und Freiheitsgrade
$n = 4 \, \text{Messungen}, \quad u = 4 \, \text{Unbekannte}, \quad f = 0 \, \text{Freiheitsgrade}$
## Fehlergleichungen
$\begin{bmatrix}
1 & 0 & -10.98 & 10.01 & 11.715 \\
0 & 1 & -10.01 & -10.98 & 9.14 \\
1 & 0 & 10.98 & -10.01 & -11.715 \\
0 & 1 & 10.01 & 10.98 & -9.14
\end{bmatrix}$
## Normalgleichungen
$\begin{bmatrix}
2.00000000 \times 10^0 & 0.00000000 \times 10^0 & 1.77635684 \times 10^{-15} & 0.00000000 \times 10^0 & 0.00000000 \times 10^0 \\
0.00000000 \times 10^0 & 2.00000000 \times 10^0 & 0.00000000 \times 10^0 & 1.77635684 \times 10^{-15} & -9.31322575 \times 10^{-10} \\
1.77635684 \times 10^{-15} & 0.00000000 \times 10^0 & 4.41521000 \times 10^2 & 6.68336497 \times 10^{-15} & -4.40244200 \times 10^2 \\
0.00000000 \times 10^0 & 1.77635684 \times 10^{-15} & 6.68336497 \times 10^{-15} & 4.41521000 \times 10^2 & 3.38199000 \times 10^1
\end{bmatrix}$
## Lösung des Gleichungssystems
$r_0 = +0.000 \, \text{m}, \quad h_0 = -0.000 \, \text{m}, \quad a = -0.997108, \quad o = +0.076599$
## Maßstab und Drehwinkel
$q = 1.0000460, \quad (q-1) = 46.0 \, \text{ppm} \, (\text{part per million}), \, [\text{mm/km}]$
$\phi = +175.607 \, \text{deg}$
## Transformierte Koordinaten von Punkt P
$r_P' = +8.406 \, \text{m}, \quad h_P' = +28.481 \, \text{m}$
## Transformierte Koordinaten von Punkt P in Gauß-Krüger
$r_{P_{gk}} = +3494079.511 \, \text{m}, \quad h_{P_{gk}} = +5795459.971 \, \text{m}$