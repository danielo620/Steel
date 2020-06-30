import numpy as np


def memgroupsec(NP, GroupShape, Shape_Set):
    for x in range(NP):
        GroupShape[x, :] = np.random.randint(Shape_Set[:, 0], high=Shape_Set[:, 1] + 1)
    return GroupShape

