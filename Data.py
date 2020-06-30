import pandas as pd
import numpy as np
import Coord
import CoordCy
import Local
import Shape
import Transformation
import os
from tempfile import TemporaryFile
import gc

Run = 'Yes'  # Yes if you want the program to run through all scripts non-stop
Path = 'C:/Users/rocky/Desktop/Code/Bridge Excel'  # Location of folder containing excel file(s)
Save = 'C:/Users/rocky/Desktop/Code/NPZ'   # Location of folder to save results for further analysis

'''















'''
with os.scandir(path=Path) as entries:
    for entry in entries:
        File = entry.name
        File = File[:-4]

        # Import excel information
        xls = pd.read_csv((os.path.join(Path, File + '.csv')), sep=',', header=1)
        Shape_Set = xls[['From', 'To']].dropna().to_numpy(dtype=np.intc) - 1
        Available_Shape = xls['Shape'].dropna().to_numpy()
        Shape_Dimension = xls[['D (in)', 't (in)', 'W (in)']].dropna(how='all').to_numpy()
        StartCord = xls[['Start X', 'Start Y', 'Start Z']].to_numpy()
        EndCord = xls[['End X', 'End Y', 'End Z']].to_numpy()
        Wt = xls['wt (lb/ft)'].dropna().to_numpy()
        LoadCase1 = xls[['wx ', 'wy ', 'wz']].dropna().to_numpy(dtype=np.single)
        LoadIdx1 = xls['Member.1'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD1 = xls[['X', 'Y', 'Z']].dropna().to_numpy()
        LoadCase2 = xls[['wx .1', 'wy .1', 'wz.1']].dropna().to_numpy(dtype=np.single)
        LoadIdx2 = xls['Member.2'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD2 = xls[['X.1', 'Y.1', 'Z.1']].dropna().to_numpy()
        LoadCase3 = xls[['wx .2', 'wy .2', 'wz.2']].dropna().to_numpy(dtype=np.single)
        LoadIdx3 = xls['Member.3'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD3 = xls[['X.2', 'Y.2', 'Z.2']].dropna().to_numpy()
        LoadCase4 = xls[['wx .3', 'wy .3', 'wz.3']].dropna().to_numpy(dtype=np.single)
        LoadIdx4 = xls['Member.4'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD4 = xls[['X.3', 'Y.3', 'Z.3']].dropna().to_numpy()
        LoadCase5 = xls[['wx .4', 'wy .4', 'wz.4']].dropna().to_numpy(dtype=np.single)
        LoadIdx5 = xls['Member.5'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD5 = xls[['X.4', 'Y.4', 'Z.4']].dropna().to_numpy()
        LoadCase6 = xls[['wx .5', 'wy .5', 'wz.5']].dropna().to_numpy(dtype=np.single)
        LoadIdx6 = xls['Member.6'].dropna().to_numpy(dtype=np.intc) - 1
        LCOORD6 = xls[['X.5', 'Y.5', 'Z.5']].dropna().to_numpy()
        Group = xls['Group'].to_numpy(dtype=np.intc) - 1
        Modules = xls[['E ', 'G']].to_numpy()
        Supports = xls[['X.6', 'Y.6', 'Z.6', 'Rx', 'Ry', 'Rz', 'Mx', 'My', 'Mz']].dropna().to_numpy()

        # Create the Member Joint Number
        AA = StartCord[:, 0] + StartCord[:, 1] ** 2 + StartCord[:, 2] ** 3 + StartCord[:, 0] ** 3 + \
            StartCord[:, 1] ** 2 + StartCord[:, 2]
        BB = EndCord[:, 0] + EndCord[:, 1] ** 2 + EndCord[:, 2] ** 3 + EndCord[:, 0] ** 3 + \
            EndCord[:, 1] ** 2 + EndCord[:, 2]
        DD = np.hstack((AA, BB))
        CC = pd.unique(DD)
        Joint = np.array([0], dtype=np.intc)  # Number of Joints
        NM = np.size(AA)  # Number of Members
        NUC = np.size(CC)  # Number of Unique Coordinates
        MPRP = np.zeros((NM, 2), dtype=np.intc)  # Member start (Mb), Member end (Me)
        COORD = np.zeros((NUC, 3), dtype=np.double)  # Unique coordinates
        Coord.Coordinate(COORD, AA, BB, CC, MPRP, StartCord, EndCord, Joint)

        # Switch axis
        C = np.zeros(np.shape(COORD))
        C[:, 0] = COORD[:, 1]
        C[:, 1] = COORD[:, 2]
        C[:, 2] = COORD[:, 0]
        COORD = C

        # Consolidate restraint coordinates
        RCC = Supports[:, 0] + Supports[:, 1] ** 2 + Supports[:, 2] ** 3 + Supports[:, 0] ** 3 + Supports[:, 1] ** 2 + \
            Supports[:, 2]
        RJoint = Supports[:, 0].size  # Number of joints with supports
        MSUP = np.zeros(Supports[:, 0:7].shape, dtype=np.intc)
        for x in range(RJoint):
            A = np.where(RCC[x] == CC)[0]
            MSUP[x, 0] = np.where(RCC[x] == CC)[0]

        # Switch axis
        xydR = Supports[:, 3:]
        MSUP[:, 1] = xydR[:, 1]
        MSUP[:, 2] = xydR[:, 2]
        MSUP[:, 3] = xydR[:, 0]
        MSUP[:, 4] = xydR[:, 4]
        MSUP[:, 5] = xydR[:, 5]
        MSUP[:, 6] = xydR[:, 3]

        # Joint Coordinate Number
        NR = np.sum(MSUP[:, 1:])  # Number of constraints degrees of freedom
        NDOF = 6 * (Joint[0]) - NR  # Number of degrees of freedom
        COORDNum = np.zeros((NUC, 6), dtype=np.intc)
        goat = MSUP
        goat = goat[np.argsort(goat[:, 0])]
        CoordCy.SupportCy(COORDNum, goat, Supports[:, 0].size, np.size(CC), NDOF)

        # Member Joint Coordinate Number
        MemberCOORDNum = np.concatenate([COORDNum[MPRP[:, 0]], COORDNum[MPRP[:, 1]]], axis=1)

        # Length of Members
        L = np.sqrt((StartCord[:, 0] - EndCord[:, 0]) ** 2 + (StartCord[:, 1] - EndCord[:, 1]) ** 2 + (
                StartCord[:, 2] - EndCord[:, 2]) ** 2)

        # Loading and Fix-End Vector
        AgrJN = np.zeros((2, 6), dtype=np.intc)
        P = np.zeros((NDOF, 6))
        LoadGoat = LoadCase1
        LoadCase1[:, 0] = LoadGoat[:, 1]
        LoadCase1[:, 1] = LoadGoat[:, 2]
        LoadCase1[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx1]
        Local.fixendforces(L[LoadIdx1], LoadCase1[:, 1], P[:, 0], Location)
        TT = LCOORD1[:, 0] + LCOORD1[:, 1] ** 2 + LCOORD1[:, 2] ** 3 + LCOORD1[:, 0] ** 3 + LCOORD1[:, 1] ** 2 + \
            LCOORD1[:, 2]
        AgrJN[0, 0] = np.where(CC == TT[0])[0]
        AgrJN[1, 0] = np.where(CC == TT[1])[0]

        LoadGoat = LoadCase2
        LoadCase2[:, 0] = LoadGoat[:, 1]
        LoadCase2[:, 1] = LoadGoat[:, 2]
        LoadCase2[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx2]
        Local.fixendforces(L[LoadIdx2], LoadCase2[:, 1], P[:, 1], Location)
        TT = LCOORD2[:, 0] + LCOORD2[:, 1] ** 2 + LCOORD2[:, 2] ** 3 + LCOORD2[:, 0] ** 3 + LCOORD2[:, 1] ** 2 + \
            LCOORD2[:, 2]
        AgrJN[0, 1] = np.where(CC == TT[0])[0]
        AgrJN[1, 1] = np.where(CC == TT[1])[0]

        LoadGoat = LoadCase3
        LoadCase3[:, 0] = LoadGoat[:, 1]
        LoadCase3[:, 1] = LoadGoat[:, 2]
        LoadCase3[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx3]
        Local.fixendforces(L[LoadIdx3], LoadCase3[:, 1], P[:, 2], Location)
        TT = LCOORD3[:, 0] + LCOORD3[:, 1] ** 2 + LCOORD3[:, 2] ** 3 + LCOORD3[:, 0] ** 3 + LCOORD3[:, 1] ** 2 + \
            LCOORD3[:, 2]
        AgrJN[0, 2] = np.where(CC == TT[0])[0]
        AgrJN[1, 2] = np.where(CC == TT[1])[0]

        LoadGoat = LoadCase4
        LoadCase4[:, 0] = LoadGoat[:, 1]
        LoadCase4[:, 1] = LoadGoat[:, 2]
        LoadCase4[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx4]
        Local.fixendforces(L[LoadIdx4], LoadCase4[:, 1], P[:, 3], Location)
        TT = LCOORD4[:, 0] + LCOORD4[:, 1] ** 2 + LCOORD4[:, 2] ** 3 + LCOORD4[:, 0] ** 3 + LCOORD4[:, 1] ** 2 + \
            LCOORD4[:, 2]
        AgrJN[0, 3] = np.where(CC == TT[0])[0]
        AgrJN[1, 3] = np.where(CC == TT[1])[0]

        LoadGoat = LoadCase5
        LoadCase5[:, 0] = LoadGoat[:, 1]
        LoadCase5[:, 1] = LoadGoat[:, 2]
        LoadCase5[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx5]
        Local.fixendforces(L[LoadIdx5], LoadCase5[:, 1], P[:, 4], Location)
        TT = LCOORD5[:, 0] + LCOORD5[:, 1] ** 2 + LCOORD5[:, 2] ** 3 + LCOORD5[:, 0] ** 3 + LCOORD5[:, 1] ** 2 + \
            LCOORD5[:, 2]
        AgrJN[0, 4] = np.where(CC == TT[0])[0]
        AgrJN[1, 4] = np.where(CC == TT[1])[0]

        LoadGoat = LoadCase6
        LoadCase6[:, 0] = LoadGoat[:, 1]
        LoadCase6[:, 1] = LoadGoat[:, 2]
        LoadCase6[:, 2] = LoadGoat[:, 0]
        Location = MemberCOORDNum[LoadIdx6]
        Local.fixendforces(L[LoadIdx6], LoadCase6[:, 1], P[:, 5], Location)
        TT = LCOORD6[:, 0] + LCOORD6[:, 1] ** 2 + LCOORD6[:, 2] ** 3 + LCOORD6[:, 0] ** 3 + LCOORD6[:, 1] ** 2 + \
            LCOORD6[:, 2]
        AgrJN[0, 5] = np.where(CC == TT[0])[0]
        AgrJN[1, 5] = np.where(CC == TT[1])[0]

        # Transformation Matrix
        Transmatrix = np.zeros((NM, 12, 12))
        Transformation.transformation(Transmatrix, EndCord[:, 1], StartCord[:, 1], EndCord[:, 2], StartCord[:, 2],
                                      EndCord[:, 0], StartCord[:, 0], L[:], NM)
        TransT = np.swapaxes(Transmatrix, 1, 2)

        # Cross-Section of each member
        NoC = np.size(Available_Shape)
        Section_Prop = np.zeros((NoC, 4))
        Shape.availableshapes(Available_Shape, Shape_Dimension, Section_Prop, NoC)

        # save data for further analysis
        BasePath = Save
        xlfileOut = TemporaryFile()
        np.savez((os.path.join(BasePath, File)), Transmatrix=Transmatrix, TransT=TransT,
                 Section_Prop=Section_Prop, L=L,
                 MemberCOORDNum=MemberCOORDNum, Shape_Set=Shape_Set, Group=Group, Modules=Modules, NM=NM, NDOF=NDOF,
                 NR=NR,
                 COORDNum=COORDNum, Wt=Wt, Shape_Dimension=Shape_Dimension, AgrJN=AgrJN, P=P)
        gc.collect()
if Run == 'Yes':
    import GA
    import Test