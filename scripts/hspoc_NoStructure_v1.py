#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from LINKS.L101 import L101
from Get_Confs_rdkit import GetConfs
from Get_mop import GetMop
import HSPOC_web_v6 as hspoc
from sklearn.preprocessing import StandardScaler
import xyz2mol


# 通过MOPAC获得分子3D结构
def GetMOPACout(SMILES, SMILESnumConfs=200, maxIters=1000):  # 待计算分子

    GetConfs(SMILES, "./temp_xyz/1.xyz", SMILESnumConfs, maxIters)
    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    charge = sum([atom.GetFormalCharge() for atom in atoms])
    if charge == 0:
        Add_command = ""
    elif charge < 0:
        Add_command = "CHARGE=" + str(charge)
    elif charge > 0:
        Add_command = "CHARGE=+" + str(charge)

    GetMop("./temp_xyz/1.xyz", "./mopac/1.mop", "PM6-DH+ OPT " + Add_command)
    mopac_process = subprocess.Popen("MOPAC2016.exe 1.mop", shell=True, cwd="./mopac/")
    mopac_process.wait()

    return


# 计算H-SPOC描述符
def GetHSPOC(
    Data,
    Solvent_onehot_Data,
    layer,
    close_layer,
    star_dis,
    close_dis,
    dis_gap,
    SMILESnumConfs,
    maxIters,
    mols=[],
    filetype="mopac",
):

    if filetype == "mopac":
        xyz_df_list = []
        calculated_smiles = []
        for i in range(len(Data)):
            DataSeries = Data.loc[i]
            SMILES = DataSeries["SMILES"]
            if SMILES not in calculated_smiles:
                calculated_smiles = calculated_smiles + [SMILES]
                GetMOPACout(SMILES, SMILESnumConfs, maxIters)
                xyz_df_list.append(L101("./mopac/1.out", SMILES))
        XYZ_Data = pd.concat(xyz_df_list, axis=0, join="outer")
        HSPOC_Data = hspoc.RUN(
            Data,
            Solvent_onehot_Data,
            XYZ_Data,
            layer,
            close_layer,
            star_dis,
            close_dis,
            dis_gap,
            mols,
        )
    elif filetype in ["xyz", "mol_object"]:
        HSPOC_Data = hspoc.RUN(
            Data,
            Solvent_onehot_Data,
            pd.DataFrame({}),
            layer,
            close_layer,
            star_dis,
            close_dis,
            dis_gap,
            mols,
        )

    return HSPOC_Data


# 根据已有模型预测分子pKa
def Predict(Data, fpname, modelname, traindafafilename, mols=[], filetype="mopac"):

    # 参数表
    layer = 3
    close_layer = 1
    star_dis = 0.3  # 单位A
    close_dis = 2.7  # 单位A
    dis_gap = 0.8  # 单位A

    SMILESnumConfs = 200
    maxIters = 1000
    Solvent_onehot_Data_filename = "./mods/Solvents_descriptor.csv"
    Solvent_onehot_Data = pd.read_csv(Solvent_onehot_Data_filename, index_col=0)

    # 读取标准化器
    with open("./mods/" + traindafafilename + ".pkl", "rb") as f:
        st = pickle.load(f)
    # 读取模型
    with open("./mods/" + modelname + ".pkl", "rb") as f:
        xgb0 = pickle.load(f)

    fp = GetHSPOC(
        Data,
        Solvent_onehot_Data,
        layer,
        close_layer,
        star_dis,
        close_dis,
        dis_gap,
        SMILESnumConfs,
        maxIters,
        mols,
        filetype,
    )
    fp.to_csv("./PredictHSPOC.csv")
    fp.replace(np.inf, 0, inplace=True)
    standard_fp = pd.read_csv("./mods/" + fpname + ".csv", low_memory=False, index_col=0)
    FP = standard_fp._append(fp, ignore_index=True)
    FP = FP.fillna(0)

    Pre_X = FP.iloc[1:, : len(standard_fp.keys())].values
    Pre_X = st.transform(Pre_X)
    Pre_y = xgb0.predict(Pre_X)

    return Pre_y


if __name__ == "__main__":

    PATH = "./Pred/"
    datafilename = "csvFileName"
    xyz_filename = ""
    sdf_filename = ""
    md_npz_filename = r""

    fpname = "fp_hspoc_v6.0_31_0.3T2.7F0.8_standard"
    modelname = "xgboost_AllTrain_fp_hspoc_v6.0_31_0.3T2.7F0.8_pKa_ibond_20240227_pKa_ibond_20240227"
    traindafafilename = "StandardScalerfp_hspoc_v6.0_31_0.3T2.7F0.8_pKa_ibond_20240227_pKa_ibond_20240227"

    mols = []
    index_list = []
    if datafilename != "":
        Data = pd.read_csv(PATH + datafilename + ".csv")
        Pre_pka = Predict(Data, fpname, modelname, traindafafilename, mols)

    if xyz_filename != "":
        for i in range(320231):
            xyz_atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(xyz_filename)
            print("Get:" + xyz_filename)
            mols.append(xyz2mol.xyz2mol(xyz_atoms, xyz_coordinates))
        Pre_pka = Predict(Data, fpname, modelname, traindafafilename, mols, filetype="mol_object")
    if sdf_filename != "":
        mols = Chem.SDMolSupplier(sdf_filename)
        Pre_pka = Predict(Data, fpname, modelname, traindafafilename, mols, filetype="mol_object")
    if md_npz_filename != "":
        data = np.load(md_npz_filename)
        Energy = data["E"]
        Force = data["F"]
        # min_Es=list(Energy[Energy>=min(Energy)])
        min_Es = [x[0] for x in Energy[:10000]]
        total_num = len(min_Es)
        index_list = [i for i in range(len(Energy)) if Energy[i][0] in min_Es]
        Data = pd.DataFrame(
            {
                "ID": list(range(total_num)),
                "SMILES": ["OC(C(C=CC=C1)=C1OC(C)=O)=O"] * total_num,
                "H_index": [13] * total_num,
                "solvent": ["H2O"] * total_num,
                "filetype": ["mol_object"] * total_num,
                "E": min_Es,
            }
        )

        zdata = data["z"]
        for Rdata in data["R"][index_list]:
            xyz_atoms, charge, xyz_coordinates = xyz2mol.read_npz_file(Rdata, zdata)
            mols = mols + xyz2mol.xyz2mol(xyz_atoms, xyz_coordinates)
        Pre_pka = Predict(Data, fpname, modelname, traindafafilename, mols, filetype="mol_object")

    Data["Pre_pKa"] = Pre_pka
    Data.to_csv(PATH + "After" + modelname + "Pred_" + datafilename + ".csv", index=0)
