import numpy as np

# Gegebene Transformationsparameter
X0 = 200.591
Y0 = -315.213
Z0 = 650.708
alpha_s = 7.86347
beta_s = 4.33383
gamma_s = -2.69604
M_ppm = 9.42703

# Maßstabsfaktor berechnen
M = 1 + M_ppm / 1e6

# Winkel in Radianten umrechnen
alpha = alpha_s / 3600 * (np.pi / 180)
beta = beta_s / 3600 * (np.pi / 180)
gamma = gamma_s / 3600 * (np.pi / 180)

# **Korrekt eingelesene Punkte im lokalen System**
punkte_lokal = np.array([
    [3683928.483, 791070.446, 5129084.753],  # Punkt 1
])

# Transformation durchführen
def transformiere_punkte(punkte, X0, Y0, Z0, alpha, beta, gamma, M):
    transformierte_punkte = np.zeros_like(punkte)
    for i, punkt in enumerate(punkte):
        x, y, z = punkt

        # Transformation
        X_neu = X0 + M * (x + y * gamma - z * beta)
        Y_neu = Y0 + M * (-x * gamma + y + z * alpha)
        Z_neu = Z0 + M * (x * beta - y * alpha + z)

        transformierte_punkte[i] = [X_neu, Y_neu, Z_neu]
    return transformierte_punkte

# Transformierte Punkte berechnen
transformierte_punkte = transformiere_punkte(punkte_lokal, X0, Y0, Z0, alpha, beta, gamma, M)

# Ergebnisse ausgeben
print("\nTransformierte Punkte:")
print(f"{'Punkt':<10}{'X (m)':<15}{'Y (m)':<15}{'Z (m)':<15}")
for i, punkt in enumerate(transformierte_punkte):
    print(f"{i + 1:<10}{punkt[0]:<15.3f}{punkt[1]:<15.3f}{punkt[2]:<15.3f}")

# Maßstabsfaktor ausgeben
print("\nMaßstab:")
print(f"M = {M:.8f}")
