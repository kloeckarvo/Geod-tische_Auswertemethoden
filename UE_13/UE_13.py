import numpy as np
import matplotlib.pyplot as plt

# Gegebene Daten: Satellitenkoordinaten und gemessene Distanzen
satellites = {
    "A": {"X": 7513858.53, "Y": 10025290.74, "Z": 23435977.16, "D": 21077225.8},
    "B": {"X": 17164353.01, "Y": 14725292.37, "Z": 13964732.93, "D": 21021306.9},
    "C": {"X": 7907487.14, "Y": 16025292.24, "Z": 19664733.99, "D": 21532371.18},
    "D": {"X": 10861899.32, "Y": 13591958.95, "Z": 19021814.35, "D": 20205032.58},
}

# Näherungskoordinaten des Empfängers
x1, y1, z1 = 4213857.00, 1025292.00, 4664733.00

# Konvergenzkriterium
tolerance = 0.001  # 1 mm
max_iterations = 100

# Iterativer Prozess
coordinates = np.array([x1, y1, z1], dtype=float)

for iteration in range(max_iterations):
    # Aktuelle Koordinaten
    x1, y1, z1 = coordinates

    # Berechnung der berechneten Distanzen
    computed_distances = []
    A = []  # Design-Matrix
    for key, data in satellites.items():
        Xs, Ys, Zs, D = data["X"], data["Y"], data["Z"], data["D"]
        computed_distance = np.sqrt((x1 - Xs)**2 + (y1 - Ys)**2 + (z1 - Zs)**2)
        computed_distances.append(computed_distance)
        # Ableitungen für die Design-Matrix
        A.append([
            (x1 - Xs) / computed_distance,
            (y1 - Ys) / computed_distance,
            (z1 - Zs) / computed_distance,
        ])

    # Residuen (Differenz zwischen gemessenen und berechneten Distanzen)
    v = np.array([satellites[key]["D"] - computed_distances[i] for i, key in enumerate(satellites)])

    # Design-Matrix
    A = np.array(A)

    # Ausgleichung nach der Methode der kleinsten Quadrate
    delta = np.linalg.inv(A.T @ A) @ A.T @ v

    # Aktualisierung der Koordinaten
    coordinates += delta

    # Abbruchbedingung
    if np.linalg.norm(delta) < tolerance:
        break

# Ergebnis: Empfängerkoordinaten
x1, y1, z1 = coordinates

# Berechnung der Standardabweichungen
residuals = v - A @ delta
sigma_squared = (residuals.T @ residuals) / (len(satellites) - 3)
cov_matrix = np.linalg.inv(A.T @ A) * sigma_squared
std_devs = np.sqrt(np.diag(cov_matrix))

# Visualisierung
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Satellitenpositionen
for key, data in satellites.items():
    ax.scatter(data["X"], data["Y"], data["Z"], label=f"Satellit {key}", s=100)
    ax.text(data["X"], data["Y"], data["Z"], f"{key}", color="blue")

# Empfängerposition
ax.scatter(x1, y1, z1, color='red', label="Empfänger", s=100)
ax.text(x1, y1, z1, "Empfänger", color="red")

# Achsenbeschriftungen
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Räumlicher Bogenschnitt: Position des Empfängers")
plt.legend()
plt.show()

# Ergebnisse anzeigen
results = {
    "Empfängerkoordinaten (X, Y, Z)": (x1, y1, z1),
    "Standardabweichungen (X, Y, Z)": std_devs
}

# Formatierte Ausgabe
receiver_coordinates = results["Empfängerkoordinaten (X, Y, Z)"]
standard_deviations = results["Standardabweichungen (X, Y, Z)"]

print("\n--- Ergebnisse des räumlichen Bogenschnitts ---")
print("Empfängerkoordinaten (Receiver Coordinates):")
print(f"  X = {receiver_coordinates[0]: .6f} m")
print(f"  Y = {receiver_coordinates[1]: .6f} m")
print(f"  Z = {receiver_coordinates[2]: .6f} m\n")

print("Standardabweichungen (Standard Deviations):")
print(f"  σ_X = {standard_deviations[0]: .6f} m")
print(f"  σ_Y = {standard_deviations[1]: .6f} m")
print(f"  σ_Z = {standard_deviations[2]: .6f} m")
print("\n------------------------------------------------")

