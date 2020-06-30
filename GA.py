import numpy as np
import Section
from joblib import Parallel, delayed
import SolutionStiffnessSpeed
import CostCalculator
import Mutation3
import UniformCrossover
import Roullette
import os
from timeit import default_timer as timer
import xlsxwriter
import pandas as pd

path = 'C:/Users/rocky/Desktop/Code/NPZ'  # path of location of data.npz folder
pathOPT = 'C:/Users/rocky/Desktop/Code/OPT'  # path to where to place information for further analyses
pathOP = 'C:/Users/rocky/Desktop/Code/Op/'  # path to where to place excel fil with optimize cross-section for VA

# Load Case
LC = 3

# Genetic Algorithm parameter
NP = 50  # number of particles
Itt = 100  # number of Iterations
PC = 1  # Ratio of children made per generation
mu = 0.035  # probability of mutating
cr = .6  # probability of crossing

# Cost function Slop
SlopD = 1.9
SlopW = 13
'''







'''
# start Parallel Pool
with Parallel(n_jobs=12, prefer="threads") as Parallel:
    with os.scandir(path=path) as entries:
        for entry in entries:

            # Extract Data from file
            File = entry.name
            npzfile = np.load((os.path.join(path, File)))
            Shape_Dimension = npzfile['Shape_Dimension']
            AgrJN = npzfile['AgrJN']
            Transmatrix = npzfile['Transmatrix']
            TransT = npzfile['TransT']
            Section_Prop = npzfile['Section_Prop']
            P = npzfile['P']
            L = npzfile['L']
            MemberCOORDNum = npzfile['MemberCOORDNum']
            G = np.min(MemberCOORDNum)
            Shape_Set = npzfile['Shape_Set']
            Group = npzfile['Group']
            NM = npzfile['NM']
            NR = npzfile['NR']
            NDOF = npzfile['NDOF']
            COORDNum = npzfile['COORDNum']
            Modules = npzfile['Modules']
            Wt = npzfile['Wt']

            # Choose from desire Load Case
            if LC == 1:
                Pf = P[:, 0]
                DNumber = AgrJN[:, 0]
            elif LC == 2:
                Pf = P[:, 1]
                DNumber = AgrJN[:, 1]
            elif LC == 3:
                Pf = P[:, 2]
                DNumber = AgrJN[:, 2]
            elif LC == 4:
                Pf = P[:, 3]
                DNumber = AgrJN[:, 3]
            elif LC == 5:
                Pf = P[:, 4]
                DNumber = AgrJN[:, 4]
            else:
                Pf = P[:, 5]
                DNumber = AgrJN[:, 5]

            # Dynamic Exploration Parameters
            nvarmin = Shape_Set[:, 0]
            nvarmax = Shape_Set[:, 1]
            sigma = (Shape_Set[:, 1] - Shape_Set[:, 0] + 1) / 2
            dynamicSig = (sigma - 1) / 100

            # Blanks for Optimization
            size = np.shape(Shape_Set[:, 0])[0]
            MemberProp = np.zeros((NP, NM, 4))
            GroupShape = np.zeros((NP, size), dtype=np.intc)
            Local_Matrix = np.zeros((NP, NM, 12, 12))
            AgrD = np.zeros(NP)
            AgrDC = np.zeros(NP)
            weight = np.zeros(NP)
            Cost = np.zeros(NP)
            CostC = np.zeros(NP)
            Agr1 = np.zeros(1)
            Agr2 = np.zeros(1)

            # Create Random finesses population
            Section.memgroupsec(NP, GroupShape, Shape_Set)
            MemberProp[:, :] = Section_Prop[GroupShape[:, Group], :]

            # start timer
            start = timer()

            # Run fitness function for starting population
            Parallel(
                delayed(SolutionStiffnessSpeed.DisplacementCy2)(NM, Modules, TransT, Transmatrix, NDOF, MemberCOORDNum,
                                                                L, Pf, MemberProp[x], Local_Matrix[x], COORDNum,
                                                                x, AgrD, DNumber, Agr1, Agr2)
                for x in range(NP))

            # evaluate starting population
            weight[:] = np.sum(Wt[GroupShape[:, Group[:]]] * L, axis=1) / 12 + SlopW  # weight function
            CostCalculator.BridgeCost(SlopD, weight, AgrD, NP, Cost)  # Cost Function
            A = np.argmin(Cost)
            BestP = Cost[A]
            W = weight[A]
            Deflection = AgrD[A]
            setT = GroupShape[A]

            for y in range(Itt):
                J = np.arange(1, NP + 1)
                J = np.flip(J)
                J = J ** 5
                Jsum = np.abs(np.sum(J))
                PP = J / Jsum
                NP = (np.round(PC * NP / 2) * 2).astype(np.intc)
                CGroup = np.zeros(GroupShape.shape, dtype=np.intc)
                chance = np.random.random(NP)

                # Elitism (Keep the best individual of the population for the next generation)
                Elite = GroupShape[0, :]
                EliteCost = Cost[0]

                # Parent Choosing and Mutation
                for z in range((NP / 2).astype(np.intc)):
                    # select parents
                    P1 = Roullette.Wheel(PP)
                    P2 = Roullette.Wheel(PP)
                    # Crossover (Create children)
                    UniformCrossover.Uniform(GroupShape[P1], GroupShape[P2], CGroup, z, chance, cr, GroupShape[0])
                    # Mutation
                    Mutation3.mutant(CGroup[2 * z], CGroup[2 * z + 1], CGroup, mu, z, sigma)

                # constrain offsprings
                CGroup[:] = np.where(CGroup > Shape_Set[:, 0], CGroup, Shape_Set[:, 0])
                CGroup[:] = np.where(CGroup < Shape_Set[:, 1], CGroup, Shape_Set[:, 1])

                # evaluate children fitness
                MemberProp[:, :] = Section_Prop[CGroup[:, Group], :]
                Parallel(
                    delayed(SolutionStiffnessSpeed.DisplacementCy2)(NM, Modules, TransT, Transmatrix, NDOF,
                                                                    MemberCOORDNum, L, Pf, MemberProp[x],
                                                                    Local_Matrix[x], COORDNum, x, AgrDC, DNumber, Agr1, Agr2)
                    for x in range(NP))

                # evaluate cost of each children
                weightC = np.zeros(NP)
                weightC[:] = np.sum(Wt[CGroup[:, Group[:]]] * (L / 12), axis=1) + SlopW

                # cost function
                CostCalculator.BridgeCost(SlopD, weightC, AgrDC, NP, CostC)
                A = np.argmin(CostC)
                BestC = CostC[A]

                # Update Population Best
                if BestC < BestP:
                    setT = CGroup[A]
                    BestP = BestC
                    W = weightC[A]
                    Deflection = AgrDC[A]
                print("Cost = ", BestP, "      AgrD = ", Deflection, "      Weight = ", W)

                # merge population
                Cost = np.hstack([Cost, CostC, EliteCost])
                X = np.argsort(Cost)
                GroupShape = np.vstack([GroupShape, CGroup, Elite])
                GroupShape = GroupShape[X, :]
                GroupShape = GroupShape[:NP, :]
                Cost = Cost[X]
                Cost = Cost[:NP]

                # dynamic mutation parameters
                mu = mu - .000005
                sigma -= dynamicSig

            # time taken to run each file
            end = timer()
            print(end - start)

            # parameters of the most fit child
            Result = Shape_Dimension[setT]
            Q = np.where(Result[:, 2] == 0)
            Result = Result.astype(np.object_)
            Result[Q, 2] = "NaN"

            # save results for further analysis
            np.savez((os.path.join(pathOPT, File[:-4] + 'OPT')), setT=setT, W=W, NDOF=NDOF, COORDNum=COORDNum, MemberCOORDNum=MemberCOORDNum, Section_Prop=Section_Prop, Group=Group, AgrJN=AgrJN, P=P, L=L, NM=NM, Modules=Modules, TransT=TransT, Transmatrix=Transmatrix)
            workbook = xlsxwriter.Workbook(pathOP + File[:-4] + '.xlsx')
            worksheet = workbook.add_worksheet()
            workbook.close()
            df = pd.DataFrame(Result)
            df.to_excel(pathOP + File[:-4] + '.xlsx', index=False)
            end = timer()
            print(end - start)


