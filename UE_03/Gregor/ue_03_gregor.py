import numpy as np
import math
import matplotlib.pyplot as plt


def manuelle_eingabe_messwerte():
    """Funktion, um Messwerte manuell einzugeben"""
    messwerte = []
    print("Gib die Messwerte ein. Tippe 'done', wenn du fertig bist.")
    while True:
        eingabe = input("Messwert: ")
        if eingabe.lower() == 'done':
            break
        try:
            wert = float(eingabe)
            messwerte.append(wert)
        except ValueError:
            print("Ungültiger Wert, bitte eine Zahl eingeben.")
    return np.array(messwerte)


def lade_messwerte_oder_manuell():
    """Funktion, um Messwerte entweder aus einer Datei zu laden oder manuell einzugeben"""
    modus = input("Möchtest du die Messwerte aus einer Datei laden (d) oder manuell eingeben (m)? ")
    if modus.lower() == 'd':
        name = input('Datei mit den Messungen = ')
        try:
            messwerte = np.loadtxt(name)
        except Exception as e:
            print(f"Fehler beim Laden der Datei: {e}")
            messwerte = manuelle_eingabe_messwerte()
    else:
        messwerte = manuelle_eingabe_messwerte()

    return messwerte


def berechne_statistik(messwerte):
    """Funktion zur Berechnung von Mittelwert und Standardabweichung sowie Vertrauensbereich"""
    N = len(messwerte)
    Mittelwert = np.mean(messwerte)
    StAbw = np.std(messwerte)
    StAbw_Mittelwert = StAbw / math.sqrt(N)
    vb_o = Mittelwert + 2 * StAbw
    vb_u = Mittelwert - 2 * StAbw
    return Mittelwert, StAbw_Mittelwert, vb_o, vb_u


def plot_ergebnisse(messwerte, Mittelwert, vb_o, vb_u):
    """Funktion zur Visualisierung der Messwerte mit Mittelwert und Vertrauensbereichen"""
    N = len(messwerte)
    nummer = np.linspace(1, N, N)

    plt.plot(nummer, messwerte, 'r', label='Messungen')
    plt.plot([0, N], [Mittelwert, Mittelwert], 'g', label='Mittelwert')
    plt.plot([0, N], [vb_o, vb_o], 'b--', label='Vertrauensbereich oben')
    plt.plot([0, N], [vb_u, vb_u], 'b--', label='Vertrauensbereich unten')

    plt.legend()
    plt.xlabel('Messung')
    plt.ylabel('Messwert')
    plt.title('Ergebnisse der Auswertung')
    plt.grid(True)
    plt.savefig("grafik.jpg")
    plt.show()


def matrizen_berechnung(messwerte, gen):
    """Funktion für die Matrixberechnungen (optional)"""
    c = 1  # Standardabweichung der Gewichtseinheit
    N = len(messwerte)
    A = np.ones([N, 1])  # Matrix A besteht aus Einsen
    P = np.zeros([N, N])

    # Matrix P initialisieren
    for i in range(N):
        P[i, i] = (c / gen[i]) ** 2

    # Berechnung des Mittelwertes über Matrizen
    Norm = np.matmul(np.matmul(np.transpose(A), P), A)
    Q = np.linalg.inv(Norm)
    HW = np.matmul(np.matmul(Q, np.transpose(A)), P)
    X = np.matmul(HW, messwerte)

    # Berechnung der Genauigkeit des Mittelwertes
    V = messwerte - np.matmul(A, X)
    s0_2 = np.matmul(np.matmul(np.transpose(V), P), V) / (N - 1)
    s0 = np.sqrt(s0_2)
    sX_2 = s0_2 * Q
    sX = np.sqrt(sX_2)

    mittelwert = X[0]
    stAbw = sX[0, 0]
    return mittelwert, stAbw


def manuelle_eingabe_standardabweichungen(N):
    """Funktion zur Eingabe von Standardabweichungen für Matrizenberechnung"""
    gen = []
    print(f"Gib die {N} Standardabweichungen ein. Tippe 'done', wenn du fertig bist.")
    for i in range(N):
        while True:
            eingabe = input(f"Standardabweichung für Messung {i + 1}: ")
            try:
                wert = float(eingabe)
                gen.append(wert)
                break
            except ValueError:
                print("Ungültiger Wert, bitte eine Zahl eingeben.")

    return np.array(gen)


def standardabweichung_auswahl(messwerte):
    """Funktion, die den Benutzer auswählt, wie die Standardabweichungen bestimmt werden"""
    modus = input("Möchtest du die Standardabweichung aus den Messwerten berechnen (b) oder manuell eingeben (m)? ")

    if modus.lower() == 'b':
        print("Standardabweichung wird aus den Messwerten berechnet.")
        StAbw = np.std(messwerte, ddof=1)  # Berechnet Standardabweichung der Messwerte
        return np.ones(len(messwerte)) * StAbw  # Uniforme Standardabweichung für alle Messwerte

    elif modus.lower() == 'm':
        print("Standardabweichungen werden manuell eingegeben.")
        return manuelle_eingabe_standardabweichungen(len(messwerte))
    else:
        print("Ungültige Auswahl. Bitte 'b' für Berechnung oder 'm' für manuelle Eingabe auswählen.")
        return standardabweichung_auswahl(messwerte)


# Hauptablauf
messwerte = lade_messwerte_oder_manuell()

# Berechnung von Mittelwert, Standardabweichung und Vertrauensbereich
Mittelwert, StAbw_Mittelwert, vb_o, vb_u = berechne_statistik(messwerte)

# Ergebnisse plotten
plot_ergebnisse(messwerte, Mittelwert, vb_o, vb_u)

# Fragen, ob Matrizenberechnungen durchgeführt werden sollen
if input("Möchtest du Matrizenberechnungen durchführen (y/n)? ").lower() == 'y':
    gen = standardabweichung_auswahl(messwerte)
    mittelwert_matrix, stAbw_matrix = matrizen_berechnung(messwerte, gen)

    # Ausgabe der Ergebnisse
    # print('Mittelwert (Matrix-Berechnung) =', np.round(mittelwert_matrix, 4), 'm')
    print('Genauigkeit des Mittelwertes (Matrix-Berechnung) =', np.round(stAbw_matrix, 4), 'm')

# Endgültige Ergebnisse speichern
np.savetxt('erg.txt', [Mittelwert, StAbw_Mittelwert])
print(f'Mittelwert = {round(Mittelwert, 3)} m')
print(f'Genauigkeit des Mittelwertes = {round(StAbw_Mittelwert, 3)} m')

