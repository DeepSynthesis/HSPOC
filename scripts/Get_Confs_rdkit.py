#!/usr/bin/python
# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd

# 通过RDKit生成若干分子构象，并通过MMFF94分子力场优化构象，选择最稳定构象作为初猜


def GetConfs(smi, Outfile, numConfs, maxIters):

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    print("Do:\t" + smi)
    AllChem.EmbedMultipleConfs(mol, randomSeed=42, numConfs=numConfs)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=maxIters)
    results = [(i, res[i][1]) for i in range(len(res)) if not res[0][0]]
    minId = results[np.argmin([x[1] for x in results])][0]

    Chem.MolToXYZFile(mol, Outfile, confId=int(minId))


if __name__ == "__main__":

    numConfs = 200
    maxIters = 1000

    PATH = "D:/LSY/HSPOC-outSample/"
    DataName = "HSPOC-outSample.csv"
    OutPATH = "D:/LSY/HSPOC-outSample/ixyz/"
    df = pd.read_csv(PATH + DataName)

    for i in range(len(df)):

        GetConfs(
            df.iloc[i]["SMILES"], OutPATH + df.iloc[i]["filename"], numConfs, maxIters
        )
