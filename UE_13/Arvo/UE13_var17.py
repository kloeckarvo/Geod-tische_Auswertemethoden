import numpy as np
import math

# Anzahl der Iterationen
Durchlaeufe = 7

# Gegebene Daten
sat_coordinates = {
    'A': [7513858.42, 10025291.04, 23435977.09],
    'B': [17164353.02, 14725292.31, 13964732.96],
    'C': [7907487.12, 16025292.25, 19664733.86],
    'D': [10861899.32, 13591958.95, 19021814.35],
}

# Näherungskoordinaten
near_coordinate = {
    'X': [4213857.00, 1025292.00, 4664733.00],  # Näherungskoordinaten
}

# Gemessene Entfernungen
measured_distances = [20942544.78, 15997583.09, 16549636.97, 16115191.61]

# Funktion zur Berechnung
def berechnung(near_coordinate):
    # 1. Berechnung von s'_iN (Abstände mit Näherungskoordinaten)
    s_prime_in = []
    for sat in sat_coordinates:
        distance = math.sqrt(
            (near_coordinate['X'][0] - sat_coordinates[sat][0])**2 +
            (near_coordinate['X'][1] - sat_coordinates[sat][1])**2 +
            (near_coordinate['X'][2] - sat_coordinates[sat][2])**2
        )
        s_prime_in.append(distance)

    # 2. Berechnung der Koeffizientenmatrix
    koeffizienten_matrix = []
    for sat_index, sat in enumerate(sat_coordinates):
        a_in = (near_coordinate['X'][0] - sat_coordinates[sat][0]) / s_prime_in[sat_index]
        b_in = (near_coordinate['X'][1] - sat_coordinates[sat][1]) / s_prime_in[sat_index]
        c_in = (near_coordinate['X'][2] - sat_coordinates[sat][2]) / s_prime_in[sat_index]
        koeffizienten_matrix.append([a_in, b_in, c_in])

    koeffizienten_matrix = np.array(koeffizienten_matrix)

    # 3. Rechte Seite des Gleichungssystems berechnen: delta_s = S_in - s'_iN
    delta_s = np.array([measured_distances[i] - s_prime_in[i] for i in range(len(measured_distances))]).reshape(-1, 1)

    # 4. Lösung des Gleichungssystems: [delta_X, delta_Y, delta_Z] using least squares
    delta_xyz, residuals, rank, s = np.linalg.lstsq(koeffizienten_matrix, delta_s, rcond=None)
    # 5. Neue Koordinaten berechnen
    new_coordinates = {
        'X': [
            near_coordinate['X'][0] + delta_xyz[0, 0],
            near_coordinate['X'][1] + delta_xyz[1, 0],
            near_coordinate['X'][2] + delta_xyz[2, 0],
        ]
    }

    # Ergebnisse ausgeben
    print("\nBerechnete Abstände s'_iN:")
    for i, distance in enumerate(s_prime_in):
        print(f"s'_iN[{i}]: {distance:.6f}")

    print("\nKoeffizientenmatrix:")
    print(koeffizienten_matrix)

    print("\nRechte Seite des Gleichungssystems (delta_s):")
    print(delta_s)

    print("\nKorrekturen (delta_X, delta_Y, delta_Z):")
    print(f"delta_X: {delta_xyz[0, 0]:.6f}")
    print(f"delta_Y: {delta_xyz[1, 0]:.6f}")
    print(f"delta_Z: {delta_xyz[2, 0]:.6f}")

    print("\nVerbesserte Koordinaten (x, y, z):")
    print(f"x: {new_coordinates['X'][0]:.6f}")
    print(f"y: {new_coordinates['X'][1]:.6f}")
    print(f"z: {new_coordinates['X'][2]:.6f}")
    return new_coordinates

# Iterative Verbesserung der Koordinaten
for i in range(Durchlaeufe):
    print(f"\nDurchlauf {i + 1}:")
    near_coordinate = berechnung(near_coordinate)