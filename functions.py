import numpy as np


def distance(YA, XA, YB, XB):
    """
    This function calculates the distance between two points (YA,XA) and (YB,XB)
    :param YA:
    :param XA:
    :param YB:
    :param XB:
    :return:
    """
    return ((YA - YB) ** 2 + (XA - XB) ** 2) ** 0.5


def gatan(YA, XA, YB, XB):
    """
    Berechnung des Richtungswinkels in Gon
    :param YA:
    :param XA:
    :param YB:
    :param XB:
    :return:
    """
    DY = YB - YA
    DX = XB - XA
    if DY > 0 and DX == 0:
        gatan = 100
    elif DY < 0 and DX == 0:
        gatan = 300
    elif DY == 0 and DX == 0:
        gatan = 'Fehler!!'
    elif DY >= 0 and DX > 0:
        gatan = np.arctan(DY / DX) * 200 / np.pi
    elif DY >= 0 and DX < 0:
        gatan = 200 + np.arctan(DY / DX) * 200 / np.pi
    elif DY < 0 and DX < 0:
        gatan = 200 + np.arctan(DY / DX) * 200 / np.pi
    else:
        gatan = 400 + np.arctan(DY / DX) * 200 / np.pi
    return gatan


def gon2grad(gon):
    """
    Umrechnung von Gon in Grad
    :param gon:
    :return:
    """
    return gon / 200 * 180


def grad2gon(grad):
    """
    Umrechnung von Grad in Gon
    :param grad:
    :return:
    """
    return grad / 180 * 200