#!/usr/bin/python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem

# 需要含空间结构的mol v3000文件，或者mopac_out由401转化的数据

from LINKS.L401 import L401
from LINKS.L405m import L405m
from LINKS.L409 import L409


#########################################################################################################
# 输入待计算数据行(pandas Series)，溶剂参数表
def Get_H_SPOC(
    DataSeries,  # 待计算数据行SMILES solvent H_index filename filetype
    distance_matrix,  # 距离矩阵
    Solvent_Data,  # 溶剂参数
    mol,  # RDkit mol
    SMILES,  #
    # 以下皆为超参
    layer,  # 结构拓扑距离
    close_layer,  # 氢键侧拓扑距离
    star_dis,  # 最小搜索距离
    close_dis,  # 最大搜索距离
    dis_gap,  # 每次搜索球壳厚度
):

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    H_index = DataSeries["H_index"]
    Solvent = DataSeries["solvent"]

    # 定义输入ID
    df0 = pd.DataFrame({"entry": ["<" + SMILES + ">" + "<" + str(H_index) + ">" + "<" + Solvent + ">"]})

    # 电荷表
    charge_data = L401(SMILES, mol)

    # 溶剂部分
    df_sol = pd.DataFrame(Solvent_Data, index=[Solvent])
    df_sol = df_sol[
        [
            "E(T)",
            "SPP",
            "SB",
            "SA",
            "Alpha",
            "Beta",
            "Pai_star",
            "Epsilon",
            "nD20 (g)",
            "Delta d",
            "Delta p",
            "Delta h",
            "Delta",
            "f(n2)",
            "Abraham_hydrogen_bond_acidity",
            "Abraham_hydrogen_bond_basicity",
            "gama(macroscopic_surface_tension)",
            "aromaticity",
            "electronegative_halogenicity",
        ]
    ]
    df_sol = df_sol.rename(index={Solvent: 0})

    # POC部分
    poc_cal = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    df_POC = poc_cal.CalcDescriptors(mol)
    df_POC = pd.DataFrame(df_POC, index=[x[0] for x in Descriptors._descList])
    df_POC = df_POC.T

    # H电荷
    df_H_Charge = pd.DataFrame({"H_Charge": [charge_data.loc[SMILES + "_GasteigerCharge_" + str(H_index), "Gasteiger_Charge"]]})

    # 描述符
    ## 获取需要计算的原子
    Descriptor_AtomSets_List = L405m(mol, H_index, layer)
    Descriptor = {}

    ## 读取直接与目标H相连的重原子A
    DH_index = list(Descriptor_AtomSets_List[1])[0]

    ## 计算键链表
    rank_matrix = np.zeros([len(atoms), len(atoms)])
    for bond in bonds:
        rank_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = float(bond.GetBondTypeAsDouble())
        rank_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = float(bond.GetBondTypeAsDouble())

    ## 非氢键部分描述
    for i in range(layer):
        Atom_Index_List = list(Descriptor_AtomSets_List[i + 1])
        for j in range(len(Atom_Index_List)):
            atom_index = int(Atom_Index_List[j])
            ### 核电荷数
            Descriptor["Radius_" + str(i + 1) + "_AtomicNum_" + str(j + 1)] = [atoms[atom_index].GetAtomicNum()]
            ### 键级
            Descriptor["Radius_" + str(i + 1) + "_bondOrder_" + str(j + 1)] = [
                max([rank_matrix[atom_index, x] for x in Descriptor_AtomSets_List[i]])
            ]
            ### Gasteiger 电荷
            Descriptor["Radius_" + str(i + 1) + "_Atom_Charge_" + str(j + 1)] = [
                charge_data.loc[SMILES + "_GasteigerCharge_" + str(atom_index), "Gasteiger_Charge"]
            ]
            ### 距离上层原子的距离
            Descriptor["Radius_" + str(i + 1) + "_dis_" + str(j + 1)] = [
                min(
                    [
                        distance_matrix[atom_index, k]
                        for k in Descriptor_AtomSets_List[i] & set([atom.GetIdx() for atom in atoms[atom_index].GetNeighbors()])
                    ]
                )
            ]
            ### 与目标H的距离
            Descriptor["Radius_" + str(i + 1) + "_disH_" + str(j + 1)] = [distance_matrix[atom_index, H_index]]
            ### 与重原子A的距离
            Descriptor["Radius_" + str(i + 1) + "_disA_" + str(j + 1)] = [distance_matrix[atom_index, DH_index]]
            ### 前述两者之比
            Descriptor["Radius_" + str(i + 1) + "_disA/disH_" + str(j + 1)] = [
                distance_matrix[atom_index, H_index] / distance_matrix[atom_index, H_index]
            ]

    ## 选择空间距离在指定值close_dis以下，同时非氢非解离H直接/相隔连接原子的原子索引
    Distance_List = []
    for atom in atoms:
        Distance_List.append(distance_matrix[H_index, atom.GetIdx()])
    Close_List = np.array(Distance_List).argsort()
    close_list = []
    close_listC = []
    for i in [star_dis + x * dis_gap for x in range(1 + int((close_dis - star_dis) / dis_gap))]:
        close_list.append(
            [
                x
                for x in Close_List
                if (distance_matrix[H_index, x] > i)
                and (distance_matrix[H_index, x] <= i + dis_gap)
                and (x != H_index)
                and (atoms[int(x)].GetAtomicNum() in [7, 8, 9, 15, 16, 17])
                and (x not in Descriptor_AtomSets_List[1])
                and (x not in Descriptor_AtomSets_List[2])
            ]
        )
        close_listC.append(
            [
                x
                for x in Close_List
                if (distance_matrix[H_index, x] > i)
                and (distance_matrix[H_index, x] <= i + dis_gap)
                and (x != H_index)
                and (atoms[int(x)].GetAtomicNum() > 1)
                and (atoms[int(x)].GetAtomicNum() not in [7, 8, 9, 15, 16, 17])
                and (x not in Descriptor_AtomSets_List[1])
                and (x not in Descriptor_AtomSets_List[2])
            ]
        )

    ## 计算满足上述条件原子的描述
    for num in range(len(close_list)):
        for i in range(len(close_list[num])):

            ### 计算余弦
            length1 = distance_matrix[H_index, close_list[num][i]]
            length2 = distance_matrix[
                [atom.GetIdx() for atom in atoms[int(H_index)].GetNeighbors()][0],
                close_list[num][i],
            ]
            length3 = distance_matrix[
                [atom.GetIdx() for atom in atoms[int(H_index)].GetNeighbors()][0],
                H_index,
            ]
            cosA = (length1 * length1 + length3 * length3 - length2 * length2) / (2 * length1 * length3)
            ### 计算距离
            distance = distance_matrix[H_index, close_list[num][i]]
            distanceA = distance_matrix[DH_index, close_list[num][i]]
            ### 核电荷数
            Descriptor["Close_" + str(i + 1) + "_AtomicNum"] = [atoms[int(close_list[num][i])].GetAtomicNum()]
            ### Gasteiger 电荷
            Descriptor["Close_" + str(i + 1) + "_Atom_Charge"] = [
                charge_data.loc[
                    SMILES + "_GasteigerCharge_" + str(close_list[num][i]),
                    "Gasteiger_Charge",
                ]
            ]
            ### 与目标H的距离
            Descriptor["Close_" + str(num + 1) + "." + str(i + 1) + "_dis"] = [distance]
            ### 与重原子A的距离
            Descriptor["Close_" + str(num + 1) + "." + str(i + 1) + "_disA"] = [distanceA]
            ### 前述两距离之比
            Descriptor["Close_" + str(num + 1) + "." + str(i + 1) + "_disA/dis"] = [distanceA / distance]
            ### 余弦值
            Descriptor["Close_" + str(num + 1) + "." + str(i + 1) + "_cos"] = [cosA]
            ### 余弦值与目标H距离之比
            Descriptor["Close_" + str(num + 1) + "." + str(i + 1) + "_cos/dis"] = [cosA / distance]

            ### 氢键部分
            Descriptor_Hbond_AtomSets_List = L405m(mol, close_list[num][i], close_layer)
            for j in range(close_layer):
                Hbond_Atom_Index_List = list(Descriptor_Hbond_AtomSets_List[j + 1])
                for k in range(len(Hbond_Atom_Index_List)):
                    hbond_atom_index = int(Hbond_Atom_Index_List[k])
                    #### 描述符含义同前
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_AtomicNum_" + str(k + 1)] = [
                        atoms[hbond_atom_index].GetAtomicNum()
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_bondOrder_" + str(j + 1)] = [
                        max([rank_matrix[hbond_atom_index, x] for x in Descriptor_Hbond_AtomSets_List[j]])
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_Atom_Charge_" + str(k + 1)] = [
                        charge_data.loc[
                            SMILES + "_GasteigerCharge_" + str(hbond_atom_index),
                            "Gasteiger_Charge",
                        ]
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_dis_" + str(k + 1)] = [
                        min(
                            [
                                (distance_matrix[hbond_atom_index, h])
                                for h in Descriptor_Hbond_AtomSets_List[j]
                                & set([atom.GetIdx() for atom in atoms[hbond_atom_index].GetNeighbors()])
                            ]
                        )
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_disH_" + str(k + 1)] = [
                        distance_matrix[H_index, hbond_atom_index]
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_disA_" + str(k + 1)] = [
                        distance_matrix[DH_index, hbond_atom_index]
                    ]
                    Descriptor["Close_" + str(i + 1) + "Radius_" + str(j + 1) + "_disA/disH_" + str(k + 1)] = [
                        distance_matrix[DH_index, hbond_atom_index] / distance_matrix[DH_index, hbond_atom_index]
                    ]

    ### 强调重原子的距离部分
    for num in range(len(close_listC)):
        for i in range(len(close_listC[num])):

            length1 = distance_matrix[H_index, close_listC[num][i]]
            length2 = distance_matrix[
                [atom.GetIdx() for atom in atoms[int(H_index)].GetNeighbors()][0],
                close_listC[num][i],
            ]
            length3 = distance_matrix[
                [atom.GetIdx() for atom in atoms[int(H_index)].GetNeighbors()][0],
                H_index,
            ]
            cosA = (length1 * length1 + length3 * length3 - length2 * length2) / (2 * length1 * length3)
            distance = distance_matrix[H_index, close_listC[num][i]]
            distanceA = distance_matrix[DH_index, close_listC[num][i]]

            Descriptor["CloseC_" + str(i + 1) + "_AtomicNum"] = [atoms[int(close_listC[num][i])].GetAtomicNum()]
            Descriptor["CloseC_" + str(i + 1) + "_Atom_Charge"] = [
                charge_data.loc[
                    SMILES + "_GasteigerCharge_" + str(close_listC[num][i]),
                    "Gasteiger_Charge",
                ]
            ]
            Descriptor["CloseC_" + str(num + 1) + "." + str(i + 1) + "_dis"] = [distance]
            Descriptor["CloseC_" + str(num + 1) + "." + str(i + 1) + "_disA"] = [distanceA]
            Descriptor["CloseC_" + str(num + 1) + "." + str(i + 1) + "_disA/dis"] = [distanceA / distance]
            Descriptor["CloseC_" + str(num + 1) + "." + str(i + 1) + "_cos"] = [cosA]
            Descriptor["CloseC_" + str(num + 1) + "." + str(i + 1) + "_cos/dis"] = [cosA / distance]

            Descriptor_Hbond_AtomSets_List = L405m(mol, close_listC[num][i], close_layer)
            for j in range(close_layer):
                Hbond_Atom_Index_List = list(Descriptor_Hbond_AtomSets_List[j + 1])
                for k in range(len(Hbond_Atom_Index_List)):
                    hbond_atom_index = int(Hbond_Atom_Index_List[k])
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_AtomicNum_" + str(k + 1)] = [
                        atoms[hbond_atom_index].GetAtomicNum()
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_bondOrder_" + str(j + 1)] = [
                        max([rank_matrix[hbond_atom_index, x] for x in Descriptor_Hbond_AtomSets_List[j]])
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_Atom_Charge_" + str(k + 1)] = [
                        charge_data.loc[
                            SMILES + "_GasteigerCharge_" + str(hbond_atom_index),
                            "Gasteiger_Charge",
                        ]
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_dis_" + str(k + 1)] = [
                        min(
                            [
                                (distance_matrix[hbond_atom_index, h])
                                for h in Descriptor_Hbond_AtomSets_List[j]
                                & set([atom.GetIdx() for atom in atoms[hbond_atom_index].GetNeighbors()])
                            ]
                        )
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_disH_" + str(k + 1)] = [
                        distance_matrix[H_index, hbond_atom_index]
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_disA_" + str(k + 1)] = [
                        distance_matrix[DH_index, hbond_atom_index]
                    ]
                    Descriptor["CloseC_" + str(i + 1) + "Radius_" + str(j + 1) + "_disA/disH_" + str(k + 1)] = [
                        distance_matrix[DH_index, hbond_atom_index] / distance_matrix[DH_index, hbond_atom_index]
                    ]

    df_Descriptor = pd.DataFrame(Descriptor)

    df = pd.concat([df0, df_sol, df_POC, df_H_Charge, df_Descriptor], axis=1, join="outer")

    return df


#########################################################################################################
# RUN
def RUN(
    Data,
    Solvent_onehot_Data,
    XYZ_Data,
    layer,
    close_layer,
    star_dis,
    close_dis,
    dis_gap,
    mols=[],
):

    df_list = []
    for i in range(len(Data)):
        ## 读取数据
        DataSeries = Data.loc[i]
        SMILES = DataSeries["SMILES"]

        ## 读取距离矩阵
        if DataSeries["filetype"] in ["mol"]:
            mol = Chem.MolFromMolFile(MolPATH + "/" + DataSeries["filename"] + ".mol", removeHs=False)
            distance_matrix = AllChem.Get3DDistanceMatrix(mol)
        elif DataSeries["filetype"] in ["mopac"]:
            mol = Chem.MolFromSmiles(SMILES)
            mol = Chem.AddHs(mol)
            distance_matrix = L409(
                mol,
                XYZ_Data[["SMILES", "Atom_Index", "x", "y", "z"]][XYZ_Data["SMILES"] == SMILES],
            )
        elif DataSeries["filetype"] in ["mol_object"]:
            mol = mols[i]
            distance_matrix = AllChem.Get3DDistanceMatrix(mols[i])

        ## 预测
        print("Generate: " + SMILES)
        df = Get_H_SPOC(
            DataSeries,
            distance_matrix,
            Solvent_onehot_Data,
            mol,
            SMILES,
            layer,
            close_layer,
            star_dis,
            close_dis,
            dis_gap,
        )
        df_list.append(df)
        print("Done: \t" + SMILES + "\n")

    ## 输出
    DF = pd.concat(df_list, axis=0, join="outer")
    DF = DF.fillna(0)
    return DF


if __name__ == "__main__":

    # parameters
    PATH = Path(__file__).parent
    Out_PATH = PATH

    Data_filename = "Pred/SAMPL6-test.csv"  # change filename here.
    Solvent_Data_filename = "mods/Solvents_descriptor.csv"  # change solvent parameters here.

    MolPATH = ""
    XYZ_Data_filename = ""

    layer = 3
    close_layer = 1
    star_dis = 0.3  # 单位A
    close_dis = 2.7  # 单位A
    dis_gap = 0.8  # 单位A

    info = str(layer) + str(close_layer) + "_" + str(star_dis) + "T" + str(close_dis) + "F" + str(dis_gap)
    Output_filename = Path(f"fp_hspoc_v6.0_{info}_{str(Data_filename)}")

    #########################################################################################################

    # Read data
    Data = pd.read_csv(PATH / Path(Data_filename))
    Solvent_Data = pd.read_csv(PATH / Path(Solvent_Data_filename), index_col=0, encoding="utf_8_sig")
    if XYZ_Data_filename != "":
        XYZ_Data = pd.read_csv(PATH / Path(XYZ_Data_filename))
    else:
        XYZ_Data = "Null"

    # Results
    DF = RUN(Data, Solvent_Data, XYZ_Data, layer, close_layer, star_dis, close_dis, dis_gap)
    DF.to_csv(Out_PATH + Output_filename, index=0)
