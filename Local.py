import numpy as np


def localmatrix(matrix, A, Iz, Iy, J, E, G, L, x):
    EE = E / (L ** 3)
    k1 = A * (L ** 2) * EE
    k2 = 12 * Iz * EE
    k3 = 6 * L * Iz * EE
    k4 = 12 * Iy * EE
    k5 = 6 * L * Iy * EE
    k6 = G * J * (L ** 2) / E * EE
    k7 = 4 * (L ** 2) * Iy * EE
    k8 = 2 * (L ** 2) * Iy * EE
    k9 = 4 * (L ** 2) * Iz * EE
    k10 = 2 * (L ** 2) * Iz * EE
    matrix[x, 0, :] = [k1, 0, 0, 0, 0, 0, -k1, 0, 0, 0, 0, 0]
    matrix[x, 1, :] = [0, k2, 0, 0, 0, k3, 0, -k2, 0, 0, 0, k3]
    matrix[x, 2, :] = [0, 0, k4, 0, -k5, 0, 0, 0, -k4, 0, -k5, 0]
    matrix[x, 3, :] = [0, 0, 0, k6, 0, 0, 0, 0, 0, -k6, 0, 0]
    matrix[x, 4, :] = [0, 0, -k5, 0, k7, 0, 0, 0, k5, 0, k8, 0]
    matrix[x, 5, :] = [0, k3, 0, 0, 0, k9, 0, -k3, 0, 0, 0, k10]
    matrix[x, 6, :] = [-k1, 0, 0, 0, 0, 0, k1, 0, 0, 0, 0, 0]
    matrix[x, 7, :] = [0, -k2, 0, 0, 0, -k3, 0, k2, 0, 0, 0, -k3]
    matrix[x, 8, :] = [0, 0, -k4, 0, k5, 0, 0, 0, k4, 0, k5, 0]
    matrix[x, 9, :] = [0, 0, 0, -k6, 0, 0, 0, 0, 0, k6, 0, 0]
    matrix[x, 10, :] = [0, 0, -k5, 0, k8, 0, 0, 0, k5, 0, k7, 0]
    matrix[x, 11, :] = [0, k3, 0, 0, 0, k10, 0, -k3, 0, 0, 0, k9]
    return matrix


def transformation(Matrix, Xe, Xb, Ye, Yb, Ze, Zb, L, x):
    MatrixR = np.zeros((3, 3))
    if np.abs(Ye - Yb) == L:
        RXy = (Ye - Yb) / L
        MatrixR[0, :] = [0, RXy, 0]
        MatrixR[1, :] = [-RXy, 0, 0]
        MatrixR[2, :] = [0, 0, 1]
        Matrix[x, :3, :3] = MatrixR
        Matrix[x, 3:6, 3:6] = MatrixR
        Matrix[x, 6:9, 6:9] = MatrixR
        Matrix[x, 9:, 9:] = MatrixR
        return Matrix
    else:
        RXx = (Xe - Xb) / L
        RXy = (Ye - Yb) / L
        RXz = (Ze - Zb) / L
        Rsq = np.sqrt(RXx ** 2 + RXz ** 2)
        MatrixR[0, :] = [RXx, RXy, RXz]
        MatrixR[1, :] = [-RXx * RXy / Rsq, Rsq, -RXy * RXz / Rsq]
        MatrixR[2, :] = [-RXz / Rsq, 0, RXx / Rsq]
        Matrix[x, :3, :3] = MatrixR
        Matrix[x, 3:6, 3:6] = MatrixR
        Matrix[x, 6:9, 6:9] = MatrixR
        Matrix[x, 9:, 9:] = MatrixR
        return Matrix


def fixendforces(L, w, Pf, Location):
    Pf[Location[:, 1]] = -w * L / 2 + Pf[Location[:, 1]]
    Pf[Location[:, 5]] = -w * (L ** 2) / 12 + Pf[Location[:, 5]]
    Pf[Location[:, 7]] = -w * L / 2 + Pf[Location[:, 7]]
    Pf[Location[:, 11]] = w * (L ** 2) / 12 + Pf[Location[:, 11]]
    return Pf


def ShapeMem(Available_Shape, Shape_Set, Shape_Dimension, Section_Prop, Group, AA):
    for x in range(0, np.size(AA)):
        if Available_Shape[Shape_Set[Group[x] - 1] - 1] == 'Round Tubing':
            D = Shape_Dimension[Shape_Set[Group[x] - 1] - 1, 0]
            t = Shape_Dimension[Shape_Set[Group[x] - 1] - 1, 1]
            Section_Prop[x, 0] = np.pi * ((D / 2) ** 2 - (D / 2 - t) ** 2)
            Section_Prop[x, 1:3] = np.pi / 4 * ((D / 2) ** 4 - (D / 2 - t) ** 4)
            Section_Prop[x, 3] = 2 * Section_Prop[x, 1]
        elif Available_Shape[Shape_Set[Group[x] - 1] - 1] == 'Rectangular Tubing':
            b = Shape_Dimension[Shape_Set[Group[x] - 1] - 1, 3]
            h = Shape_Dimension[Shape_Set[Group[x] - 1] - 1, 0]
            t = Shape_Dimension[Shape_Set[Group[x] - 1] - 1, 1]
            bi = (b - 2 * t)
            hi = (h - 2 * t)
            Section_Prop[x, 0] = b * h - bi * hi
            Section_Prop[x, 1] = b * (h ** 3) / 12 - bi * (hi ** 3) / 12
            Section_Prop[x, 3] = h * (b ** 3) / 12 - hi * (bi ** 3) / 12
            Section_Prop[x, 2] = 2 * (b ** 2) * (h ** 2) / (b / t + h / t)
    return Section_Prop
