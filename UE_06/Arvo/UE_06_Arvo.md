# Abgabe Übung 06
> **Arvo Klöck | 909003**  
> **06.11.2024**
## Funktionsaufrufe:
```python   
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
```
## print_passpunkte(passpunkte):  
**Beschreibung:**  
Diese Funktion druckt eine Tabelle der gegebenen Passpunkte mit ihren Koordinaten im Quell- und Zielsystem.  
**Mathematische Beschreibung:**  
Keine spezifische mathematische Berechnung, nur Formatierung und Ausgabe der Daten.

```python
def print_passpunkte(passpunkte):
    headers = ["#", "Name", "y", "x", "Y", "X"]
    table = [
        [i+1, name, f"{yy:+10.3f} m", f"{xx:+10.3f} m", f"{YY:+10.3f} m", f"{XX:+10.3f} m"]
        for i, (name, yy, xx, YY, XX) in enumerate(passpunkte)
    ]
    print(f"-- Gegeben: {len(passpunkte)} Passpunkte im yx-/Quell- und YX-/Zielsystem")
    print(tabulate(table, headers, tablefmt="grid"))
    print()
```
```
Gegeben: 5 Passpunkte im yx-/Quell- und YX-/Zielsystem
+-----+--------+-----------+-----------+---------------+----------------+  
|   # | Name   | y         | x         | Y             | X              |  
+=====+========+===========+===========+===============+================+  
|   1 | A      | +1.560 m  | +30.590 m | +494086.687 m | +5795436.910 m |  
+-----+--------+-----------+-----------+---------------+----------------+  
|   2 | B      | +46.900 m | +22.530 m | +494043.548 m | +5795420.795 m |  
+-----+--------+-----------+-----------+---------------+----------------+  
|   3 | C      | +23.820 m | +2.010 m  | +494052.974 m | +5795450.206 m |  
+-----+--------+-----------+-----------+---------------+----------------+  
|   4 | D      | +71.890 m | +5.450 m  | +494013.342 m | +5795422.786 m |  
+-----+--------+-----------+-----------+---------------+----------------+  
|   5 | E      | +35.430 m | +34.170 m | +494059.338 m | +5795416.599 m |  
+-----+--------+-----------+-----------+---------------+----------------+
```  

## berechne_schwerpunkte(passpunkte):  
**Beschreibung:**  
Berechnet die Schwerpunkte der Passpunkte sowohl im Quell- als auch im Zielsystem.  
**Mathematische Beschreibung:**  
$$( yS = \frac{1}{n} \sum_{i=1}^{n} y_i )$$
$$( xS = \frac{1}{n} \sum_{i=1}^{n} x_i )$$
$$( YS = \frac{1}{n} \sum_{i=1}^{n} Y_i )$$
$$( XS = \frac{1}{n} \sum_{i=1}^{n} X_i )$$

```python
def berechne_schwerpunkte(passpunkte):
    yS = np.mean([yy for _, yy, _, _, _ in passpunkte])
    xS = np.mean([xx for _, _, xx, _, _ in passpunkte])
    YS = np.mean([YY for _, _, _, YY, _ in passpunkte])
    XS = np.mean([XX for _, _, _, _, XX in passpunkte])
    print(f"Schwerpunktkoordinaten: yS={yS:+10.3f} m, xS={xS:+10.3f} m; YS={YS:+10.3f} m, XS={XS:+10.3f} m")
    print()
    return yS, xS, YS, XS
```

```
Schwerpunktkoordinaten: 
yS=   +35.920 m 
xS=   +18.950 m
YS=   +494051.178 m
XS=+5795429.459 m
```
## erstelle_matrizen(passpunkte, yS, xS, YS, XS):  
**Beschreibung:**  
Erstellt die Koeffizientenmatrix ( A ) und die Beobachtungsvektoren ( b ) für das Gleichungssystem.  
**Mathematische Beschreibung:**   
$$( A )$$ ist eine $$( 2n \times 4 )$$ Matrix, die aus den transformierten Koordinaten der Passpunkte besteht.
$$( b )$$ ist ein $$( 2n \times 1 )$$ Vektor, der die Differenzen der Koordinaten der Passpunkte zu den Schwerpunkten enthält.

```python
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
```

```
Fehlergleichungen
[[ 35.5092  -7.4508   1.       0.     -34.36  ]
 [  7.4508  35.5092   0.       1.      11.64  ]
 [ -7.6298   8.6642   1.       0.      10.98  ]
 [ -8.6642  -7.6298   0.       1.       3.58  ]
 [  1.7962 -20.7468   1.       0.     -12.1   ]
 [ 20.7468   1.7962   0.       1.     -16.94  ]
 [-37.8358   6.6732   1.       0.      35.97  ]
 [ -6.6732 -37.8358   0.       1.     -13.5   ]
 [  8.1602  12.8602   1.       0.      -0.49  ]
 [-12.8602   8.1602   0.       1.      15.22  ]]
```

## berechne_normalgleichungen(A, b):  
**Beschreibung:**  
Berechnet die Normalgleichungen für das Gleichungssystem.  
**Mathematische Beschreibung:**    
$$( N = A^T A )$$
$$( y = A^T b )$$

```python
def berechne_normalgleichungen(A, b):
    N = A.T @ A
    y = A.T @ b
    print("-- Normalgleichungen")
    print(np.hstack((N, y)))
    print()
    return N, y
```

```

Normalgleichungen
[
[ 3.59140893e+03  1.35878721e-14 -5.82076609e-11  3.72529030e-09 -3.09194292e+03]
[ 1.35878721e-14  3.59140893e+03 -3.72529030e-09 -5.82076609e-11 1.82647851e+03]
[-5.82076609e-11 -3.72529030e-09  5.00000000e+00  0.00000000e+00 -7.10542736e-15]
[ 3.72529030e-09 -5.82076609e-11  0.00000000e+00  5.00000000e+00 7.10542736e-15]
 ]

```
## loese_gleichungssystem(N, y):  
**Beschreibung:**  
Löst das Gleichungssystem der Normalgleichungen und berechnet die Transformationsparameter.  
**Mathematische Beschreibung:**  
$$( Q = N^{-1} )$$
$$( \text{params} = N^{-1} y )$$
Die Parameter $$( a ), ( b ), ( Ty ), ( Tx )$$ werden aus dem Vektor params extrahiert.

```python
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
```

```
Transformationsparameter:
  a (cos) = -0.860928
  b (sin) = +0.508569
  Ty = +0.000 m
  Tx = +0.000 m

```
## berechne_massstab_und_drehwinkel(a, b):  
**Beschreibung:**  
Berechnet den Maßstab und den Drehwinkel der Transformation.    
**Mathematische Beschreibung:**    
$$( s = \sqrt{a^2 + b^2} )$$
$$( sppm = (s - 1) \times 10^6 )$$
$$( \theta = \arctan\left(\frac{b}{a}\right) \times \frac{180}{\pi} )$$

```python
def berechne_massstab_und_drehwinkel(a, b):
    s = sqrt(a**2 + b**2)
    sppm = (s - 1) * 1.0E6
    theta = atan2(b, a) * 180 / pi
    print(f"Skalierung       s={s:12.7f}, (s-1)={sppm:.1f} ppm (part per million)")
    print(f"Drehwinkel     theta={theta:+10.3f} deg")
    print()
    return s, theta
```

```
Skalierung       s=   0.9999192, (s-1)=-80.8 ppm (part per million)
Drehwinkel       theta=  +149.429 deg

```
## transformiere_punkt(neupunkt, yS, xS, s, theta, Ty, Tx):
**Beschreibung:**  
Transformiert die Koordinaten eines neuen Punktes basierend auf den berechneten Transformationsparametern.  
**Mathematische Beschreibung:**   
$$( yP = yP - yS )$$
$$( xP = xP - xS )$$
$$( YP' = Tx + s \left( \cos(\theta) yP - \sin(\theta) xP \right) )$$
$$( XP' = Ty + s \left( \sin(\theta) yP + \cos(\theta) xP \right) )$$
```python
def transformiere_punkt(neupunkt, yS, xS, s, theta, Ty, Tx):
    yP, xP = neupunkt[1] - yS, neupunkt[2] - xS
    YP_trans = Tx + s * (cos(theta * pi / 180) * yP - sin(theta * pi / 180) * xP)
    XP_trans = Ty + s * (sin(theta * pi / 180) * yP + cos(theta * pi / 180) * xP)
    print(f"Transformierte Koordinaten von Punkt P: YP'={YP_trans:+15.3f} m, XP'={XP_trans:+15.3f} m")
    print()
    return YP_trans, XP_trans
```

```
Transformierte Koordinaten von Punkt P:
YP'=        -12.574 m
XP'=         +6.429 m
```
## berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS):  
**Beschreibung:** 
Berechnet die Gauß-Krüger-Koordinaten des transformierten Punktes.  
**Mathematische Beschreibung:**  
$$( YP_{gk} = YS + YP' )$$
$$( XP_{gk} = XS + XP' )$$

```python
def berechne_gauss_krueger_koordinaten(YP_trans, XP_trans, YS, XS):
    YP_gk = YS + YP_trans
    XP_gk = XS + XP_trans
    print(f"Transformierte Koordinaten von Punkt P in Gauß-Krüger: YP_gk={YP_gk:+15.3f} m, XP_gk={XP_gk:+15.3f} m")
    print()
    return YP_gk, XP_gk
```

```
Transformierte Koordinaten von Punkt P in Gauß-Krüger:
YP_gk=    +494038.604 m
XP_gk=   +5795435.888 m
```