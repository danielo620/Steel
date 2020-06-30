import numpy as np


def Coordinate(COORD, AA, BB, CC, Mbe, StartCord, EndCord, Joint):
    for x in range(0, np.size(CC)):
        Start = np.where(CC[x] == AA)
        End = np.where(CC[x] == BB)
        Mbe[Start, 0] = Joint[0]
        Mbe[End, 1] = Joint[0]
        if Start[0].size > 0:
            COORD[x, :] = StartCord[np.amin(Start), :]
        else:
            COORD[x, :] = EndCord[np.amin(End), :]
        Joint[0] = Joint[0] + 1
    return COORD, Joint
