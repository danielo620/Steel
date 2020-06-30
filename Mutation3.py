import numpy as np


def mutant(C1, C2, CGroup, mu, z, sigma):
    Loc1 = np.where(np.random.random(np.size(CGroup[z])) < mu)
    Loc2 = np.where(np.random.random(np.size(CGroup[z])) < mu)
    R1 = np.random.randn(np.size(Loc1))
    R2 = np.random.randn(np.size(Loc2))
    CGroup[z * 2, Loc1] = C1[Loc1] + sigma[Loc1] * R1
    CGroup[z * 2 + 1, Loc2] = C2[Loc2] + sigma[Loc2] * R2
    return CGroup
