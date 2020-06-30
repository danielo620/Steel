import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from Test2 import localmatrix2
import StiffnessNewCyth


def DisplacementCy2(NM, Modules, TransT, Transmatrix, NDOF, MemberCOORDNum, L, Pf,
                    MemberProp, Local_Matrix, COORDNum, x, AgrD, DNumber, Agr1, Agr2):

    # Local Stiffness Matrix
    Local_Matrix = localmatrix2(Local_Matrix, MemberProp[:, 0], MemberProp[:, 1], MemberProp[:, 2],
                                MemberProp[:, 3], Modules[:, 0], Modules[:, 1], L)

    # Global Stiffness Matrix
    Global_Matrix = np.einsum('ijk,ikl ->ijl', TransT, Local_Matrix)
    Global_Matrix = np.einsum('ijk,ikl ->ijl', Global_Matrix, Transmatrix)

    # Stiffness Matrix
    Number = NM * 12 * 12
    row = np.zeros(Number, dtype=np.intc)
    col = np.zeros(Number, dtype=np.intc)
    data = np.zeros(Number)
    StiffnessNewCyth.NewStiff(NDOF - 1, Global_Matrix, NM, MemberCOORDNum, row, col, data)

    # Solve for joint Displacements
    ST = csr_matrix((data, (row, col)), shape=(NDOF, NDOF))
    displacement = spsolve(ST, Pf)
    Agr1[0] = displacement[COORDNum[DNumber[0], 1]]
    Agr2[0] = displacement[COORDNum[DNumber[1], 1]]
    AgrD[x] = displacement[COORDNum[DNumber[0], 1]] + displacement[COORDNum[DNumber[1], 1]]
    return displacement, Agr1, Agr2
