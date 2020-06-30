import numpy as np


def Wheel(P):
    r = np.random.random() * np.sum(P)
    C = np.cumsum(P)
    A = np.argmax(r <= C)
    return A
