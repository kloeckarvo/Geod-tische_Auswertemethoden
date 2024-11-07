import numpy as np
from math import sqrt, atan2, pi, cos, sin
from tabulate import tabulate

def print_passpunkte(passpunkte):
    headers = ["#", "Name", "y", "x", "Y", "X"]
    table = [
        [i+1, name, f"{yy:+10.3f} m", f"{xx:+10.3f} m", f"{YY:+10.3f} m", f"{XX:+10.3f} m"]
        for i, (name, yy, xx, YY, XX) in enumerate(passpunkte)
    ]
    print(f"-- Gegeben: {len(passpunkte)} Passpunkte im yx-/Quell- und YX-/Zielsystem")
    print(tabulate(table, headers, tablefmt="grid"))
    print()

def berechne_schwerpunkte(passpunkte):
    yS = np.mean([yy for _, yy, _, _, _ in passpunkte])
    xS = np.mean([xx for _, _, xx, _, _ in passpunkte])
    YS = np.mean([YY for _, _, _, YY, _ in passpunkte])
    XS = np.mean([XX for _, _, _, _, XX in passpunkte])
    print(f"Schwerpunktkoordinaten: yS={yS:+10.3f} m, xS={xS:+10.3f} m; YS={YS:+10.3f} m, XS={XS:+10.3f} m")
    print()
    return yS, xS, YS, XS

def erstelle_matrizen(passpunkte, yS, xS, YS, XS):
    n = len(passpunkte)
    A = np.zeros((2 * n, 4))
    b = np.zeros((2 * n, 1))
    for i, (name, yy, xx, YY, XX) in enumerate(passpunkte):
        yy, xx, YY, XX = yy - yS, xx - xS, YY - YS, XX - XS
        A[2 * i] = [YY, -XX, 1, 0]
        A[2 * i + 1] = [XX, YY, 0, 1]
        b[2 * i] = [yy]
        b[2 * i + 1] = [xx]
    print("-- Fehlergleichungen")
    print(np.hstack((A, b)))
    print()
    return A, b

def berechne_normalgleichungen(A, b):
    N = A.T @ A
    y = A.T @ b
    print("-- Normalgleichungen")
    print(np.hstack((N, y)))
    print()
    return N, y

def loese_gleichungssystem(N, y):
    Q = np.linalg.inv(N)
    params = np.linalg.solve(N, y).flatten()
    a, b, Ty, Tx = params
    print(f"Transformationsparameter:")
    print(f"  a (cos) = {a:+.6f}")
    print(f"  b (sin) = {b:+.6f}")
    print(f"  Ty = {Ty:+.3f} m")
    print(f"  Tx = {Tx:+.3f} m")
    return a, b, Ty, Tx

def berechne_massstab_und_drehwinkel(a, b):
    s = sqrt(a**2 + b**2)
    sppm = (s - 1) * 1.0E6
    theta = atan2(b, a) * 180 / pi
    print(f"Skalierung       s={s:12.7f}, (s-1)={sppm:.1f} ppm (part per million)")
    print(f"Drehwinkel     theta={theta:+10.3f} deg")
    print()
    return s, theta

def transformiere_punkt(neupunkt, yS, xS, s, theta, Ty, Tx):
    yP, xP = neupunkt[1] - yS, neupunkt[2] - xS
    YP_trans = Tx + s * (cos(theta * pi / 180) * yP - sin(theta * pi / 180) * xP)
    XP_trans = Ty + s * (sin(theta * pi / 180) * yP + cos(theta * pi / 180) * xP)
    print(f"Transformierte Koordinaten von Punkt P: YP'={YP_trans:+15.3f} m, XP'={XP_trans:+15.3f} m")
    print()
    return YP_trans, XP_trans

def berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS):
    YP_gk = YS + YP_trans
    XP_gk = XS + XP_trans
    print(f"Transformierte Koordinaten von Punkt P in Gauß-Krüger: YP_gk={YP_gk:+15.3f} m, XP_gk={XP_gk:+15.3f} m")
    print()
    return YP_gk, XP_gk

def berechne_restklaffungen(A, a, b, Ty, Tx, passpunkte):
    v = A @ np.array([[a], [b], [Ty], [Tx]]) - b
    table = [
        [i+1, passpunkt[0], f"{-v[2 * i, 0] * 1000:+6.1f} mm", f"{-v[2 * i + 1, 0] * 1000:+6.1f} mm"]
        for i, passpunkt in enumerate(passpunkte)
    ]
    print("-- Restklaffungen (in mm)")
    print(tabulate(table, headers=["#", "Name", "vy", "vx"], tablefmt="grid"))

def berechne_redundanzanteile(A, Q):
    R = np.diag(A @ Q @ A.T)
    table = [[i+1, r] for i, r in enumerate(R)]
    print("\nRedundanzanteile:")
    print(tabulate(table, headers=["#", "Redundanzanteil"], tablefmt="grid"))

def main():
    passpunkte = [
        ("A", 1.560, 30.590, 494086.687, 5795436.910),
        ("B", 46.900, 22.530, 494043.548, 5795420.795),
        ("C", 23.820, 2.010, 494052.974, 5795450.206),
        ("D", 71.890, 5.450, 494013.342, 5795422.786),
        ("E", 35.430, 34.170, 494059.338, 5795416.599),
    ]
    neupunkt = ("P", 50.017, 19.810)

    print_passpunkte(passpunkte)
    yS, xS, YS, XS = berechne_schwerpunkte(passpunkte)
    A, b = erstelle_matrizen(passpunkte, yS, xS, YS, XS)
    N, y = berechne_normalgleichungen(A, b)
    a, b, Ty, Tx = loese_gleichungssystem(N, y)
    s, theta = berechne_massstab_und_drehwinkel(a, b)
    YP_trans, XP_trans = transformiere_punkt(neupunkt, yS, xS, s, theta, Ty, Tx)
    berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS)
    berechne_restklaffungen(A, a, b, Ty, Tx, passpunkte)
    Q = np.linalg.inv(N)
    berechne_redundanzanteile(A, Q)

if __name__ == "__main__":
    main()