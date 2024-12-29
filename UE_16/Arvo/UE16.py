import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def create_dataframes():
    identische_punkte = {
        "Nr": ['A', 'B', 'C'],
        "y": [0.312, 45.652, 22.572],
        "x": [29.342, 21.282, 0.762],
        "Y": [494087.123, 494043.984, 494053.418],
        "X": [5795438.615, 5795422.500, 5795451.911],
        "sY": [0.010, 0.020, 0.010],
        "sX": [0.010, 0.020, 0.010],
    }

    neue_punkte = {
        "Nr": ['P'],
        "y": [50.017],
        "x": [19.810],
    }

    ident_df = pd.DataFrame(identische_punkte)
    neu_df = pd.DataFrame(neue_punkte)
    return ident_df, neu_df

def residuals(params, y, x, Y, X):
    a1, a2, a3, b1, b2, b3 = params
    X_calc = a1 * y + a2 * x + a3
    Y_calc = b1 * y + b2 * x + b3
    return np.hstack(((X_calc - X), (Y_calc - Y)))

def transform_points(y, x, params):
    a1, a2, a3, b1, b2, b3 = params
    X_trans = a1 * y + a2 * x + a3
    Y_trans = b1 * y + b2 * x + b3
    return X_trans, Y_trans

def perform_least_squares(y_vals, x_vals, Y_vals, X_vals):
    initial_params = np.zeros(6)
    result = least_squares(residuals, initial_params, args=(y_vals, x_vals, Y_vals, X_vals), method='lm')
    return result.x

def calculate_residuals(ident_df, params):
    ident_df["X_calc"], ident_df["Y_calc"] = transform_points(ident_df["y"], ident_df["x"], params)
    ident_df["v_X"] = ident_df["X"] - ident_df["X_calc"]
    ident_df["v_Y"] = ident_df["Y"] - ident_df["Y_calc"]
    return ident_df

def transform_new_points(neu_df, params):
    neu_df["X"], neu_df["Y"] = transform_points(neu_df["y"], neu_df["x"], params)
    return neu_df

def plot_transformation(ident_df, neu_df):
    plt.figure(figsize=(10, 6))
    scatter_original = plt.scatter(ident_df["X"], ident_df["Y"], color="orange", label="Passpunkte (Original)", marker="o")
    scatter_calculated = plt.scatter(ident_df["X_calc"], ident_df["Y_calc"], color="purple", label="Passpunkte (Berechnet)", marker="x")
    scatter_new = plt.scatter(neu_df["X"], neu_df["Y"], color="blue", label="Neuer Punkt (Transformiert)", marker="^")

    for i in range(len(ident_df)):
        plt.plot([ident_df["X"].iloc[i], ident_df["X_calc"].iloc[i]],
                 [ident_df["Y"].iloc[i], ident_df["Y_calc"].iloc[i]],
                 linestyle="--", color="gray")

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Visualisierung der Transformation: Passpunkte und neuer Punkt", fontsize=14)
    plt.legend(handles=[scatter_original, scatter_calculated, scatter_new], loc="best")
    plt.grid()
    plt.show()

def main():
    ident_df, neu_df = create_dataframes()
    y_vals = ident_df["y"].values
    x_vals = ident_df["x"].values
    Y_vals = ident_df["Y"].values
    X_vals = ident_df["X"].values

    params = perform_least_squares(y_vals, x_vals, Y_vals, X_vals)
    ident_df = calculate_residuals(ident_df, params)
    neu_df = transform_new_points(neu_df, params)

    print("Residuen der Passpunkte:")
    print(ident_df[["Nr", "v_X", "v_Y"]])

    plot_transformation(ident_df, neu_df)

    print("Transformierte neuer Punkt:")
    print(neu_df)

if __name__ == "__main__":
    main()