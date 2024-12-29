import numpy as np

def dist(YA, XA, YB, XB):
    return ((YA - YB)**2 + (XA - XB)**2)**0.5

def initialize_matrices():
    A = np.zeros([5, 4])
    B = np.zeros([5, 1])
    X = np.transpose(np.array([1, 1, 1, 1]))
    return A, B, X

def update_matrices(A, B, koord, s13, s14, s12, s23, s24):
    s13n = dist(koord[0, 1], koord[0, 2], koord[2, 1], koord[2, 2])
    s14n = dist(koord[0, 1], koord[0, 2], koord[3, 1], koord[3, 2])
    s12n = dist(koord[0, 1], koord[0, 2], koord[1, 1], koord[1, 2])
    s23n = dist(koord[1, 1], koord[1, 2], koord[2, 1], koord[2, 2])
    s24n = dist(koord[1, 1], koord[1, 2], koord[3, 1], koord[3, 2])

    A[0, 0] = -(koord[2, 1] - koord[0, 1]) / s13n
    A[0, 1] = -(koord[2, 2] - koord[0, 2]) / s13n
    B[0] = s13 - s13n

    A[1, 0] = -(koord[3, 1] - koord[0, 1]) / s14n
    A[1, 1] = -(koord[3, 2] - koord[0, 2]) / s14n
    B[1] = s14 - s14n

    A[2, 0] = -(koord[1, 1] - koord[0, 1]) / s12n
    A[2, 1] = -(koord[1, 2] - koord[0, 2]) / s12n
    A[2, 2] = (koord[1, 1] - koord[0, 1]) / s12n
    A[2, 3] = (koord[1, 2] - koord[0, 2]) / s12n
    B[2] = s12 - s12n

    A[3, 2] = -(koord[2, 1] - koord[1, 1]) / s23n
    A[3, 3] = -(koord[2, 2] - koord[1, 2]) / s23n
    B[3] = s23 - s23n

    A[4, 2] = -(koord[3, 1] - koord[1, 1]) / s24n
    A[4, 3] = -(koord[3, 2] - koord[1, 2]) / s24n
    B[4] = s24 - s24n

    return A, B

def calculate_corrections(A, B):
    AT = np.transpose(A)
    Norm = np.matmul(AT, A)
    Q = np.linalg.inv(Norm)
    X = np.matmul(np.matmul(Q, AT), B)
    return X, Q

def update_coordinates(koord, X):
    koord[0, 1] += X[0].item()
    koord[0, 2] += X[1].item()
    koord[1, 1] += X[2].item()
    koord[1, 2] += X[3].item()
    return koord

def calculate_accuracy(A, X, B, Q):
    V = np.matmul(A, X) - B
    s0_2 = np.matmul(np.transpose(V), V) / 1
    s0 = np.sqrt(s0_2)
    sy1 = s0 * np.sqrt(Q[0, 0])
    sx1 = s0 * np.sqrt(Q[1, 1])
    sy2 = s0 * np.sqrt(Q[2, 2])
    sx2 = s0 * np.sqrt(Q[3, 3])
    return sy1, sx1, sy2, sx2

def main():
    koord = np.array([
        [1, 3494070.00, 5795260.00],
        [2, 3494260.00, 5795255.00],
        [3, 3494082.82, 5795440.63],
        [4, 3494259.39, 5795448.35]
    ])

    s13, s14, s12, s23, s24 = 181.874, 268.886, 193.304, 258.029, 193.999
    Grenze = 0.00001
    A, B, X = initialize_matrices()
    Anz = 0

    while abs(X[0]) > Grenze or abs(X[1]) > Grenze:
        A, B = update_matrices(A, B, koord, s13, s14, s12, s23, s24)
        X, Q = calculate_corrections(A, B)
        koord = update_coordinates(koord, X)
        Anz += 1

        if Anz > 5:
            print('Anzahl der Iterationen ist zu groß!')
            break

    sy1, sx1, sy2, sx2 = calculate_accuracy(A, X, B, Q)

    print('NN ', ' Y ', ' X ', ' sY ', ' sX ')
    print('1 ', np.round(koord[0, 1], 3), np.round(koord[0, 2], 3), np.round(sy1, 4), np.round(sx1, 4))
    print('2 ', np.round(koord[1, 1], 3), np.round(koord[1, 2], 3), np.round(sy2, 4), np.round(sx2, 4))

if __name__ == "__main__":
    main()