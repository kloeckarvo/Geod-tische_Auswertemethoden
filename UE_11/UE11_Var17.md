# Aufgabe 11
>Bearbeitet von:  
> Arvo Klöck  | 909003


## Code
```python
import numpy as np

# Einlesen
passp_a = np.array([
    [1, 3683928.483, 790070.446, 5126084.753],
    [2, 3686951.401, 792458.630, 5129085.569],
    [3, 3685917.563, 789992.468, 5131081.688]
])

passp_n = np.array([
    [1, 3683641.509, 789485.448, 5125706.191],
    [2, 3686664.494, 791873.410, 5128706.990],
    [3, 3685630.629, 789407.261, 5130703.019]
])

# Matrizen bilden
grosse_a = np.shape(passp_a)
grosse_n = np.shape(passp_n)

if grosse_a == grosse_n:
    N = grosse_a[0]
else:
    print('Anzahl der Passp. überprüfen!')
    exit()

A = np.zeros([3 * N, 7])
B = np.zeros([3 * N, 1])

# Zeilen X
i = 0
k = 0
while i < 3 * N:
    A[i, 0] = 1
    A[i, 1] = 0
    A[i, 2] = 0
    A[i, 3] = 0
    A[i, 4] = -passp_a[k, 3]
    A[i, 5] = passp_a[k, 2]
    A[i, 6] = passp_a[k, 1]
    B[i, 0] = passp_n[k, 1]
    i += 3
    k += 1

# Zeilen Y
i = 1
k = 0
while i < 3 * N:
    A[i, 0] = 0
    A[i, 1] = 1
    A[i, 2] = 0
    A[i, 3] = passp_a[k, 3]
    A[i, 4] = 0
    A[i, 5] = -passp_a[k, 1]
    A[i, 6] = passp_a[k, 2]
    B[i, 0] = passp_n[k, 2]
    i += 3
    k += 1

# Zeilen Z
i = 2
k = 0
while i < 3 * N:
    A[i, 0] = 0
    A[i, 1] = 0
    A[i, 2] = 1
    A[i, 3] = -passp_a[k, 2]
    A[i, 4] = passp_a[k, 1]
    A[i, 5] = 0
    A[i, 6] = passp_a[k, 3]
    B[i, 0] = passp_n[k, 3]
    i += 3
    k += 1

# Berechnung von Transf.Parametern
Norm = np.transpose(A) @ A
Q = np.linalg.inv(Norm)
HW = Q @ np.transpose(A)
X = HW @ B
X0 = X[0]
Y0 = X[1]
Z0 = X[2]
M = X[6]
alfa = X[3] / M
betta = X[4] / M
gamma = X[5] / M

# Umrechnung und Ausgabe von Transformationsparametern
M_ppm = (1 - M) * 10**6
alfa_s = alfa * (180 / np.pi) * 3600
betta_s = betta * (180 / np.pi) * 3600
gamma_s = gamma * (180 / np.pi) * 3600

print('X0 (m) = ', np.round(X0, 3))
print('Y0 (m) = ', np.round(Y0, 3))
print('Z0 (m) = ', np.round(Z0, 3))
print('Alfa (") = ', np.round(alfa_s, 3))
print('Betta (") = ', np.round(betta_s, 3))
print('Gamma (") = ', np.round(gamma_s, 3))
print('Maßstab (ppm) = ', np.round(M_ppm, 4))

# Berechnung von Restklaffungen
neup = np.array([
    [1, 3683928.483, 790070.446, 5126084.753],
    [2, 3686951.401, 792458.630, 5129085.569],
    [3, 3685917.563, 789992.468, 5131081.688]
])

grosse_np = np.shape(neup)
erg = np.zeros([grosse_np[0], 4])

i = 0
while i < grosse_np[0]:
    erg[i, 0] = neup[i, 0]
    erg[i, 1] = X0.item() + M.item() * (neup[i, 1] + neup[i, 2] * gamma.item() - neup[i, 3] * betta.item())
    erg[i, 2] = Y0.item() + M.item() * (-neup[i, 1] * gamma.item() + neup[i, 2] + neup[i, 3] * alfa.item())
    erg[i, 3] = Z0.item() + M.item() * (neup[i, 1] * betta.item() - neup[i, 2] * alfa.item() + neup[i, 3])
    i += 1

test = erg - passp_n

# Ausgabe
# np.savetxt('neu_transf.txt', erg)

print(erg)
print(test)
```


Ergebnisse:
```text
X0 (m) =  [-327.134]
Y0 (m) =  [-306.517]
Z0 (m) =  [-268.919]
Alfa (") =  [-6.218]
Betta (") =  [-2.976]
Gamma (") =  [6.248]
Maßstab (ppm) =  [-15.6678]
[[1.00000000e+00 3.68364151e+06 7.89485447e+05 5.12570619e+06]
 [2.00000000e+00 3.68666450e+06 7.91873412e+05 5.12870699e+06]
 [3.00000000e+00 3.68563063e+06 7.89407260e+05 5.13070301e+06]]
[[ 0.          0.00430338 -0.00067238 -0.0020826 ]
 [ 0.          0.0055678   0.00189308 -0.00371981]
 [ 0.          0.00286691 -0.00132567 -0.00442008]]
```