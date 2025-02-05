#!/usr/bin/python
# -*- coding: utf-8 -*-

# 依据L101中的xyz表输出距离矩阵的部分值

import numpy as np
import pandas as pd
from rdkit import Chem


def L409(mol, df):

    atoms = mol.GetAtoms()
    total_num = len(atoms)
    xyz = np.array([np.array(df[df["Atom_Index"] == i][["x", "y", "z"]]) for i in range(total_num)])
    xyz = xyz.reshape(total_num, 3).astype(float)
    O_X = np.matrix(xyz[:, 0])
    O_Y = np.matrix(xyz[:, 1])
    O_Z = np.matrix(xyz[:, 2])
    D2 = np.square(O_X - O_X.T) + np.square(O_Y - O_Y.T) + np.square(O_Z - O_Z.T)
    D = np.sqrt(D2)

    return D
