import numpy as np
from math import sqrt, pi, cos, sin, atan2

def berechne_schwerpunkte(passpunkte):
    yS = np.mean([yy for _, yy, _, _, _ in passpunkte])
    xS = np.mean([xx for _, _, xx, _, _ in passpunkte])
    YS = np.mean([YY for _, _, _, YY, _ in passpunkte])
    XS = np.mean([XX for _, _, _, _, XX in passpunkte])
    return yS, xS, YS, XS

def erstelle_matrizen(passpunkte, yS, xS, YS, XS):
    n = len(passpunkte)
    A = np.zeros((2 * n, 6))
    b = np.zeros((2 * n, 1))
    for i, (name, yy, xx, YY, XX) in enumerate(passpunkte):
        yy, xx, YY, XX = yy - yS, xx - xS, YY - YS, XX - XS
        A[2 * i] = [yy, xx, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, yy, xx, 1]
        b[2 * i] = [YY]
        b[2 * i + 1] = [XX]
    return A, b

def berechne_normalgleichungen(A, b):
    N = A.T @ A
    y = A.T @ b
    return N, y

def loese_gleichungssystem(N, y):
    Q = np.linalg.inv(N)
    params = np.linalg.solve(N, y).flatten()
    return params

def transformiere_punkt(neupunkt, yS, xS, params):
    a1, b1, Ty, a2, b2, Tx = params
    yP, xP = neupunkt[1] - yS, neupunkt[2] - xS
    YP_trans = a1 * yP + b1 * xP + Ty
    XP_trans = a2 * yP + b2 * xP + Tx
    return YP_trans, XP_trans

def berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS):
    YP_gk = YS + YP_trans
    XP_gk = XS + XP_trans
    return YP_gk, XP_gk

def berechne_restklaffungen(A, params, b):
    v = A @ params.reshape(-1, 1) - b
    return v

def berechne_mittlerer_punktfehler(v, n):
    num_observations = 2 * n
    num_parameters = 6
    f = num_observations - num_parameters
    sigma0_squared = (v.T @ v)[0, 0] / f
    sigma0 = sqrt(sigma0_squared)
    return sigma0

def berechne_redundanzanteile(A, Q):
    R = np.diag(A @ Q @ A.T)
    return R

def main():
    passpunkte = [
        ("A", 1.560, 30.590, 494086.687, 5795436.910),
        ("B", 46.900, 22.530, 494043.548, 5795420.795),
        ("C", 23.820, 2.010, 494052.974, 5795450.206),
        ("D", 71.890, 5.450, 494013.342, 5795422.786),
        ("E", 35.430, 34.170, 494059.338, 5795416.599),
    ]
    neupunkt = ("P", 50.017, 19.810)

    yS, xS, YS, XS = berechne_schwerpunkte(passpunkte)
    A, b = erstelle_matrizen(passpunkte, yS, xS, YS, XS)
    N, y = berechne_normalgleichungen(A, b)
    params = loese_gleichungssystem(N, y)
    YP_trans, XP_trans = transformiere_punkt(neupunkt, yS, xS, params)
    YP_gk, XP_gk = berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS)
    v = berechne_restklaffungen(A, params, b)
    sigma0 = berechne_mittlerer_punktfehler(v, len(passpunkte))
    R = berechne_redundanzanteile(A, np.linalg.inv(N))

    print(f"Transformierte Koordinaten von Punkt P: YP'={YP_trans:+15.3f} m, XP'={XP_trans:+15.3f} m")
    print(f"Transformierte Koordinaten von Punkt P in Gauß-Krüger: YP_gk={YP_gk:+15.3f} m, XP_gk={XP_gk:+15.3f} m")
    print(f"Mittlerer Punktfehler (sigma_0): {sigma0:.3f} mm")
    print("\nRedundanzanteile:")
    for i, r in enumerate(R):
        print(f"{i+1:02d}: {r:.4f}")

    print("\nFehlergleichungen (v):")
    for i, vi in enumerate(v.flatten()):
        print(f"v[{i+1}] = {vi:.6f}")

    print("\nNormalgleichungen (N):")
    print(N)

    print("\nRechte Seite der Normalgleichungen (y):")
    print(y)

if __name__ == "__main__":
    main()