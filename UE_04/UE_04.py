# Import
import numpy as np
# Statische Messwerte
# Statische Messwerte
mess = np.array([
    [202.17, 102.713, 0.005],
    [204.04, 102.712, 0.003],
    [209.98, 102.707, 0.003],
    [213.52, 102.710, 0.005],
    [214.89, 102.711, 0.005],
    [217.01, 102.702, 0.003],
    [220.33, 102.702, 0.003],
    [225.06, 102.700, 0.003],
    [227.88, 102.701, 0.003],
    [233.01, 102.699, 0.004],
    [234.22, 102.703, 0.005]
])
c = 0.003 #Standardabweichung der Gewichtseinheit
# Matrizen bilden
grosse = np.shape(mess)
N = grosse[0]
A = np.zeros ([N, 2])
P = np.zeros ([N, N])H
B = np.zeros ([N, 1])
i = 0
while i < N:
  A[i, 0] = 1
  A[i, 1] = mess[i, 0]
  P[i, i] = (c / mess[i, 2])**2
  B[i, 0] = mess[i, 1]
  i += 1
# Berechnung von Transformationsparametern
Norm = np.matmul(np.matmul(np.transpose(A), P), A)
Q = np.linalg.inv(Norm)
HW = np.matmul(np.matmul(Q, np.transpose(A)), P)
X = np.matmul(HW, B)
# Genauigkeit
Pol = np.matmul (A, X)
V = B - Pol
s0_2 = np.matmul(np.matmul(np.transpose(V), P), V) / (N - 2)
s0 = np.sqrt(s0_2)
sb = s0*np.sqrt(Q[0, 0])
sm = s0*np.sqrt(Q[1, 1])
# Grafik
import matplotlib.pyplot as plt
plt.plot(mess[:, 0], mess[:, 1], 'g', linewidth = 2, label = 'Messwerte')
plt.plot(mess[:, 0], Pol[:, 0], '--r', linewidth = 2, label = 'Polynom')
plt.xlabel('Zeit'), plt.ylabel('Messwerte')
plt.title('Regressionsanalyse')
plt.grid('True')
# Ausgabe
print('b = ', np.round(X[0, 0], 2),' sb = ', np.round(sb[0, 0], 3))
print('m = ', np.round(X[1, 0], 4), ' sm = ', np.round(sm[0, 0], 4))