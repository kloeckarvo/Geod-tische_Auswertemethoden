import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Daten aus der Aufgabe
identische_punkte = {
    "Nr": [1, 2, 3, 4, 5, 6],
    "y": [4.247, 42.723, 89.875, 68.523, 23.485, 79.199],
    "x": [15.045, 66.376, 57.860, 2.136, 40.710, 29.998],
    "Y": [23.781, 23.766, 73.782, 73.770, 23.779, 73.770],
    "X": [23.771, 93.759, 93.771, 23.761, 53.269, 53.164],
    "sY": [0.005, 0.005, 0.005, 0.005, 0.010, 0.010],
    "sX": [0.005, 0.005, 0.005, 0.005, 0.010, 0.010],
}

neue_punkte = {
    "Nr": [11, 12, 13, 14],
    "y": [33.688, 46.624, 66.482, 56.052],
    "x": [37.464, 56.662, 53.011, 33.187],
}

# Konvertierung in DataFrames
ident_df = pd.DataFrame(identische_punkte)
neu_df = pd.DataFrame(neue_punkte)

# Funktion zur Berechnung der Residuen (Ausgleichsrechnung)
def residuals(params, y, x, Y, X):
    a1, a2, a3, b1, b2, b3, c1, c2 = params
    denom = c1 * y + c2 * x + 1
    X_calc = (a1 * y + a2 * x + a3) / denom
    Y_calc = (b1 * y + b2 * x + b3) / denom
    return np.hstack(((X_calc - X), (Y_calc - Y)))

# Funktion zur Transformation der Punkte
def transform_points(y, x, params):
    a1, a2, a3, b1, b2, b3, c1, c2 = params
    denom = c1 * y + c2 * x + 1
    X_trans = (a1 * y + a2 * x + a3) / denom
    Y_trans = (b1 * y + b2 * x + b3) / denom
    return X_trans, Y_trans

# Initiale Schätzung der Parameter
initial_params = np.zeros(8)

# Input-Daten für die Ausgleichung
y_vals = ident_df["y"].values
x_vals = ident_df["x"].values
Y_vals = ident_df["Y"].values
X_vals = ident_df["X"].values

# Gewichtung basierend auf den Standardabweichungen
weights = 1 / np.hstack((ident_df["sX"].values, ident_df["sY"].values))

# Least-Squares-Ausgleichung
result = least_squares(residuals, initial_params, args=(y_vals, x_vals, Y_vals, X_vals), method='lm')

# Gewonnene Parameter
params = result.x

# Transformation der Passpunkte
ident_df["X_calc"], ident_df["Y_calc"] = transform_points(ident_df["y"], ident_df["x"], params)

# Residuenberechnung (v = Beobachtet - Berechnet)
ident_df["v_X"] = ident_df["X"] - ident_df["X_calc"]
ident_df["v_Y"] = ident_df["Y"] - ident_df["Y_calc"]

# Ausgabe der Residuen
print("Residuen der Passpunkte:")
print(ident_df[["Nr", "v_X", "v_Y"]])

# Transformation der neuen Punkte
neu_df["X"], neu_df["Y"] = transform_points(neu_df["y"], neu_df["x"], params)

# Visualisierung der Transformation
plt.figure(figsize=(10, 6))

# Passpunkte (Original) plotten
scatter_original = plt.scatter(ident_df["X"], ident_df["Y"], color="green", label="Passpunkte (Original)", marker="o")

# Passpunkte (Berechnet) plotten
scatter_calculated = plt.scatter(ident_df["X_calc"], ident_df["Y_calc"], color="blue", label="Passpunkte (Berechnet)", marker="x")

# Neue Punkte (Transformiert) plotten
scatter_new = plt.scatter(neu_df["X"], neu_df["Y"], color="red", label="Neue Punkte (Transformiert)", marker="^")

# Verbindungslinien zwischen Original und Berechnet für Passpunkte
for i in range(len(ident_df)):
    plt.plot([ident_df["X"].iloc[i], ident_df["X_calc"].iloc[i]],
             [ident_df["Y"].iloc[i], ident_df["Y_calc"].iloc[i]],
             linestyle="--", color="gray")

# Achsenbeschriftung und Titel
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.title("Visualisierung der Transformation: Passpunkte und neue Punkte", fontsize=14)

# Legende erstellen
plt.legend(handles=[scatter_original, scatter_calculated, scatter_new], loc="best")

# Gitter anzeigen
plt.grid()
plt.show()


# Ausgabe der transformierten neuen Punkte
print("Transformierte neue Punkte:")
print(neu_df)

