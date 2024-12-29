import numpy as np
import matplotlib.pyplot as plt

def calculate_receiver_coordinates(satellites, initial_coordinates, tolerance=0.001, max_iterations=100):
    coordinates = np.array(initial_coordinates, dtype=float)

    for iteration in range(max_iterations):
        x1, y1, z1 = coordinates
        computed_distances = []
        A = []

        for key, data in satellites.items():
            Xs, Ys, Zs, D = data["X"], data["Y"], data["Z"], data["D"]
            computed_distance = np.sqrt((x1 - Xs)**2 + (y1 - Ys)**2 + (z1 - Zs)**2)
            computed_distances.append(computed_distance)
            A.append([
                (x1 - Xs) / computed_distance,
                (y1 - Ys) / computed_distance,
                (z1 - Zs) / computed_distance,
            ])

        v = np.array([satellites[key]["D"] - computed_distances[i] for i, key in enumerate(satellites)])
        A = np.array(A)
        delta = np.linalg.inv(A.T @ A) @ A.T @ v
        coordinates += delta

        if np.linalg.norm(delta) < tolerance:
            break

    return coordinates, A, v

def calculate_standard_deviations(A, v, delta, num_satellites):
    residuals = v - A @ delta
    sigma_squared = (residuals.T @ residuals) / (num_satellites - 3)
    cov_matrix = np.linalg.inv(A.T @ A) * sigma_squared
    std_devs = np.sqrt(np.diag(cov_matrix))
    return std_devs

def plot_results(satellites, receiver_coordinates):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for key, data in satellites.items():
        ax.scatter(data["X"], data["Y"], data["Z"], label=f"Satellit {key}", s=100)
        ax.text(data["X"], data["Y"], data["Z"], f"{key}", color="blue")

    x1, y1, z1 = receiver_coordinates
    ax.scatter(x1, y1, z1, color='red', label="Empfänger", s=100)
    ax.text(x1, y1, z1, "Empfänger", color="red")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Räumlicher Bogenschnitt: Position des Empfängers")
    plt.legend()
    plt.show()

def main():
    satellites = {
        "A": {"X": 7513858.42, "Y": 10025291.04, "Z": 23435977.09, "D": 21077225.89},
        "B": {"X": 17164353.02, "Y": 14725292.31, "Z": 13964732.96, "D": 21021306.93},
        "C": {"X": 7907487.12, "Y": 16025292.25, "Z": 19664733.86, "D": 21532371.13},
        "D": {"X": 10861899.32, "Y": 13591958.95, "Z": 19021814.35, "D": 20205032.62},
    }

    initial_coordinates = [4213857.00, 1025292.00, 4664733.00]
    receiver_coordinates, A, v = calculate_receiver_coordinates(satellites, initial_coordinates)
    std_devs = calculate_standard_deviations(A, v, receiver_coordinates - initial_coordinates, len(satellites))

    plot_results(satellites, receiver_coordinates)

    print("\n--- Ergebnisse des räumlichen Bogenschnitts ---")
    print("Empfängerkoordinaten (Receiver Coordinates):")
    print(f"  X = {receiver_coordinates[0]: .6f} m")
    print(f"  Y = {receiver_coordinates[1]: .6f} m")
    print(f"  Z = {receiver_coordinates[2]: .6f} m\n")

    print("Standardabweichungen (Standard Deviations):")
    print(f"  σ_X = {std_devs[0]: .6f} m")
    print(f"  σ_Y = {std_devs[1]: .6f} m")
    print(f"  σ_Z = {std_devs[2]: .6f} m")
    print("\n------------------------------------------------")

if __name__ == "__main__":
    main()