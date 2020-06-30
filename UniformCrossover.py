import numpy as np


def Uniform(P1, P2, CGroup, z, chance, cr, best):
    k = np.size(P1)
    j = np.random.randint(2, size=k)
    if z == 0:
        CGroup[z * 2, :] = best
    else:
        if chance[2 * z] <= cr:
            CGroup[z * 2, :] = P1 * j + (1 - j) * P2
        else:
            CGroup[z * 2, :] = P1

    if chance[2 * z + 1] <= cr:
        CGroup[z * 2 + 1, :] = P2 * j + (1 - j) * P1
    else:
        CGroup[z * 2 + 1, :] = P2
    return CGroup
