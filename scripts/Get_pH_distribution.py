#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hspoc_NoStructure_v1 import Predict
import time


#
def pH_plot(a, name, a_true=np.array([0])):

    color_list = plt.cm.brg(np.linspace(0.1, 0.9, len(a[0, :])))
    plt.figure(figsize=(10, 5))
    plt.title("Molecule Distribution")
    plt.ylabel("Distribution")
    plt.xlabel("pH")
    for i in range(len(a[0, :])):
        plt.plot([x / 10 for x in range(141)], a[:, i], c=color_list[i])
    if np.sum(a_true) != 0:
        for i in range(len(a_true[0, :])):
            plt.plot([x / 10 for x in range(141)], a_true[:, i], c=color_list[i], linestyle="--")
    plt.savefig(name + ".png", dpi=300, bbox_inches="tight")
    return


#
if __name__ == "__main__":

    fpname = "fp_hspoc_v6.0_31_0.3T2.7F0.8_noGly2_standard"
    modelname = "XGBoost_AllTrain_fp_hspoc_v6.0_31_0.3T2.7F0.8_pKa_ibond_20240227_noGly2_pKa_ibond_20240227_noGly2"
    traindatafilename = "StandardScalerfp_hspoc_v6.0_31_0.3T2.7F0.8_pKa_ibond_20240227_noGly2_pKa_ibond_20240227_noGly2"

    PATH = "./Pred/"
    datafilename = "Gly_states"
    filetype = "mopac"
    True_pKa = True

    mols = []
    # 输入数据按解离顺序排序
    Data = pd.read_csv(PATH + datafilename + ".csv")
    Pre_pKa = Predict(Data, fpname, modelname, traindatafilename, mols, filetype=filetype)
    print(Pre_pKa)
    start = time.time()
    Data["Pre_pKa"] = Pre_pKa
    Data.to_csv(PATH + "After" + modelname + "PM6Pred_" + datafilename + ".csv", index=0)
    # 作图
    a = np.array([[0 for i in range(len(Pre_pKa) + 1)]])
    c_calc = lambda x, i: np.power(10, i * x - sum(Pre_pKa[:i]))
    for grid in range(141):
        pH = grid / 10
        c = np.array([1] + [c_calc(pH, i + 1) for i in range(len(Pre_pKa))])
        a = np.append(a, [np.array([x / np.sum(c) for x in c])], axis=0)
    print(a)
    #
    a_true = np.array([[0, 0, 0]])
    if True_pKa:
        pKa = Data["pKa"].values.tolist()
        a_true = np.array([[0 for i in range(len(pKa) + 1)]])
        ct_calc = lambda x, i: np.power(10, i * x - sum(pKa[:i]))
        for grid in range(141):
            pH = grid / 10
            c = np.array([1] + [ct_calc(pH, i + 1) for i in range(len(pKa))])
            a_true = np.append(a_true, [np.array([x / np.sum(c) for x in c])], axis=0)
    #
    pH_plot(a[1:, :], "Gly_Pred", a_true[1:, :])

    end = time.time()
    print(end - start)
