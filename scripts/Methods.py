#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn import tree, svm, neighbors, ensemble, gaussian_process
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.tree import DecisionTreeRegressor
import xgboost
import lightgbm as lgb
import catboost
import pickle


# 读取数据
def load_data(data_name, fp_name, y_name, split_ratio, random_seed=42):

    data = pd.read_csv(data_name + ".csv")
    fingerprint = pd.read_csv(fp_name + ".csv", low_memory=False, index_col=0)
    fingerprint = fingerprint.replace(np.inf, 0)
    fingerprint = fingerprint.replace(np.nan, 0)

    y = data[y_name].values
    X = fingerprint.values

    st = StandardScaler()

    X = st.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=random_seed)
    return X_train, X_test, y_train, y_test, X, y, st


# xgboost算法
def xgb(X_train, y_train):
    title = r"Extra Tree Regressor"
    xgb1 = xgboost.XGBRegressor(
        learning_rate=0.04,
        n_estimators=1200,
        max_depth=10,
        min_child_weight=7,
        subsample=0.8,
        colsample_bytree=0.6,
        gamma=0.1,
        reg_alpha=2,
        reg_lambda=0.5,
        verbosity=2,
        n_jobs=-1,
        seed=42,
    )
    xgb1.fit(X_train, y_train)

    return xgb1


# 对结果绘图
def plotResult(
    y,
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    split_ratio,
    fp,
    info,
    figure_title="",
    method_name="",
):

    rms_train = (np.mean((y_train - y_pred_train) ** 2)) ** 0.5
    rms_test = (np.mean((y_test - y_pred_test) ** 2)) ** 0.5
    print("rms_train: %s; r^2_train: %s" % (round(rms_train, 3), round(r2_score(y_train, y_pred_train), 3)))
    print("rms_test: %s; r^2_test: %s" % (round(rms_test, 3), round(r2_score(y_test, y_pred_test), 3)))

    # plot
    plt.figure(figsize=(5, 3))
    plt.scatter(y_train, y_pred_train, label="Train", c="deepskyblue", alpha=0.5, s=20)
    plt.title(figure_title)
    plt.xlabel("Experimence")
    plt.ylabel("Prediction")
    x_start = min(y) - (max(y) - min(y)) * 0.1
    x_end = min(y) + (max(y) - min(y)) * 1.1
    plt.xlim((x_start, x_end))
    plt.ylim((x_start, x_end))
    plt.scatter(y_test, y_pred_test, c="navy", label="Test", alpha=0.5, s=20)
    x1 = [x_start, x_end]
    y1 = x1
    plt.plot(x1, y1, c="lightcoral", alpha=0.8)
    plt.legend(loc=4)
    #
    point_x1 = min(y) - (max(y) - min(y)) * 0.05
    #
    test1 = "RMSE = " + str(round(rms_test, 3))
    test2 = "R$^2$ = " + str(round(r2_score(y_test, y_pred_test), 3))
    point_y3 = (max(y) - min(y)) * 1 + min(y)
    point_y4 = (max(y) - min(y)) * 0.9 + min(y)
    plt.text(point_x1, point_y3, test1, weight="light")
    plt.text(point_x1, point_y4, test2, weight="light")
    #
    plt.savefig(
        method_name + "_" + info + str(split_ratio) + "train_" + fp + ".png",
        dpi=300,
        bbox_inches="tight",
    )
    return


# 保存模型
def modelGeneration(m0, st, info, method_name=""):

    with open(method_name + "_AllTrain_" + info + ".pkl", "wb") as f:
        pickle.dumps(m0, f)
    with open("StandardScaler" + info + ".pkl", "wb") as f:
        pickle.dumps(st, f)

    return


# k折交叉验证
def kFold(X, y, k, fp_name, info="", random_seed=42, method_name=""):

    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    R2 = []
    RMSE = []
    MAE = []
    for train_index, test_index in kf.split(X):
        xgb_model = xgb(X[train_index], y[train_index])
        prediction = xgb_model.predict(X[test_index])
        actuals = y[test_index]
        R2.append(r2_score(actuals, prediction))
        MAE.append(mean_absolute_error(prediction, actuals))
        RMSE.append(np.sqrt(mean_squared_error(prediction, actuals)))

    print("R2_avg=" + str(np.mean(R2)) + "\n" + "MAE_avg=" + str(np.mean(MAE)) + "\n" + "RMSE_avg=" + str(np.mean(RMSE)))
    with open(method_name + "_" + info + str(k) + "FoldResult_" + fp_name + ".txt", "w") as f:
        f.write("R2_avg=" + str(np.mean(R2)) + "\n" + str(R2) + "\n")
        f.write("MAE_avg=" + str(np.mean(MAE)) + "\n" + str(MAE) + "\n")
        f.write("RMSE_avg=" + str(np.mean(RMSE)) + "\n" + str(RMSE) + "\n")

    return np.mean(R2), np.mean(MAE), np.mean(RMSE)


# 参数重要性分析
def Contribution_Analyst(fp, xgb_model, info="", method_name=""):

    fingerprint = pd.read_csv(fp + ".csv", low_memory=False, index_col=0)
    im = pd.DataFrame({"parameter": fingerprint.columns, "importance": xgb_model.feature_importances_})
    im = im.sort_values(by="importance", ascending=False)
    im.to_csv(method_name + "_" + info + "contribution_" + fp + ".csv")

    return im


# 树回归方法
def tree_reg(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Tree Regression"
    tree_reg = tree.DecisionTreeRegressor(max_depth=40)
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, tree_reg, m)


# 随机森林方法
def random_forest(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Random Forest"
    rf = RandomForestRegressor(
        n_estimators=1000,
        criterion="squared_error",
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        verbose=1,
    )
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, rf, m)


# 高斯过程方法
def gpr(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Gaussian Process"
    kernel = 1.0 * RBF(length_scale=1) + WhiteKernel(noise_level=1)
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, gp, m)


# 支持向量机方法
def svr(X_train, X_test, y_train, y_test, X, y, m):
    title = r"SVR"
    svr = svm.SVR(kernel="rbf", C=1e8, gamma=0.01)
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, svr, m)


# k临近方法
def knn(X_train, X_test, y_train, y_test, X, y, m):
    title = r"KNN"
    n_neighbors = 8
    weights = "distance"
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, knn, m)


# ada boost方法
def ada(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Ada Boost"
    ada = ensemble.AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=200),
        n_estimators=200,
        random_state=42,
        learning_rate=0.01,
    )
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, ada, m)


# 梯度boost方法
def gbrt(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Gradient Boosting"
    gbrt = ensemble.GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=16,
        loss="squared_error",
        random_state=42,
    )
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, gbrt, m)


# 贪婪回归方法
def bag_r(X_train, X_test, y_train, y_test, X, y, m):
    title = r"Bagging Regressor"
    bag_r = BaggingRegressor()
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, bag_r, m)


# 额外树方法
def tree_extra(X_train, X_test, y_train, y_test, X, y, m):
    from sklearn.tree import ExtraTreeRegressor

    title = r"Extra Tree Regressor"
    tree_extra = ExtraTreeRegressor()
    regressor_method(X_train, X_test, y_train, y_test, X, y, title, tree_extra, m)


# 运行前述方法
def regressor_method(X_train, X_test, y_train, y_test, X, y, title, regressor, m):

    regressor.fit(X_train, y_train)
    score = round(regressor.score(X_test, y_test), 3)
    # ^2, rms
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    rms_train = (np.mean((y_train - y_pred_train) ** 2)) ** 0.5
    rms_test = (np.mean((y_test - y_pred_test) ** 2)) ** 0.5
    # print result
    print("RMS_train", rms_train)
    print("r^2 score_train", r2_score(y_train, y_pred_train))
    print("RMS_test", rms_test)
    print("r^2 score_test", r2_score(y_test, y_pred_test))
    # plot
    plt.subplot(m)
    plt.scatter(y_train, y_pred_train, label="Train", c="blue", alpha=0.5)
    score = r" (Score = " + str(score) + r")"
    plt.title(title + score)
    plt.ylabel("Predicted pKa")
    x_start = min(y) - (max(y) - min(y)) * 0.1
    x_end = min(y) + (max(y) - min(y)) * 1.1
    plt.xlim((x_start, x_end))
    plt.ylim((x_start, x_end))
    plt.scatter(y_test, y_pred_test, c="lightgreen", label="Test", alpha=0.5)
    x1 = [x_start, x_end]
    y1 = x1
    plt.plot(x1, y1, c="lightcoral", alpha=0.8)
    plt.legend(loc=4)
    #
    test5 = "RMS_train = " + str(round(rms_train, 3))
    test6 = "r^2 score_train = " + str(round(r2_score(y_train, y_pred_train), 3))
    point_x1 = min(y) - (max(y) - min(y)) * 0.05
    point_y1 = (max(y) - min(y)) * 1 + min(y)
    point_y2 = (max(y) - min(y)) * 0.89 + min(y)
    plt.text(
        point_x1,
        point_y1,
        test5,
        weight="light",
        bbox=dict(facecolor="blue", alpha=0.2),
    )
    plt.text(
        point_x1,
        point_y2,
        test6,
        weight="light",
        bbox=dict(facecolor="blue", alpha=0.2),
    )
    #
    test1 = "RMS_test = " + str(round(rms_test, 3))
    test2 = "r^2 score_test = " + str(round(r2_score(y_test, y_pred_test), 3))
    point_y3 = (max(y) - min(y)) * 0.76 + min(y)
    point_y4 = (max(y) - min(y)) * 0.65 + min(y)
    plt.text(
        point_x1,
        point_y3,
        test1,
        weight="light",
        bbox=dict(facecolor="lightgreen", alpha=0.2),
    )
    plt.text(
        point_x1,
        point_y4,
        test2,
        weight="light",
        bbox=dict(facecolor="lightgreen", alpha=0.2),
    )
    #
    plt.tight_layout()


# lightGBM方法
def lightGBM_method(X, y):

    params = {
        "learning_rate": 0.05,
        "random_state": 42,
    }

    lgb_model = lgb.LGBMRegressor(**params)
    lgb_model.fit(X, y)

    return lgb_model


# catboost方法
def catboost_method(X, y):

    params = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "loss_function": "RMSE",
        "depth": 10,
        "random_seed": 42,
    }

    cat_model = catboost.CatBoostRegressor(**params)
    cat_model.fit(X, y)

    return cat_model


if __name__ == "__main__":

    ###############################################################################
    data_name = "pKa_ibond_20240227"
    fp_name = [
        "fp_hspoc_v6.0_31_0.3T2.7F0.8_noH_pKa_ibond_20240227",
    ]
    figure_title = ""
    split_ratio = 0.8
    ###############################################################################

    for fp in fp_name:

        X_train, X_test, y_train, y_test, X, y, st = load_data(data_name, fp, "pKa", split_ratio)

        # 输出模型
        xgb0 = xgb(X, y)
        modelGeneration(xgb0, st, fp + "_" + data_name, "XGBoost")

        model = xgb(X_train, y_train)
        # model = lightGBM_method(X_train, y_train)
        # model = catboost_method(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        plotResult(
            y,
            y_train,
            y_pred_train,
            y_test,
            y_pred_test,
            split_ratio,
            fp,
            info="",
            figure_title=figure_title,
            method_name="xgboost",
        )
        # Contribution_Analyst(fp,model,method_name='lgb')
        # kFold(X,y,5,fp,info='',method_name='xgboost')

    # for fp in fp_name:
    # X_train, X_test, y_train, y_test, X, y, st = load_data(data_name, fp, 'pKa', split_ratio)
    # plt.figure(figsize=(20,10))
    # regression
    # tree_reg(X_train, X_test, y_train, y_test, X, y, 331)
    # random_forest(X_train, X_test, y_train, y_test, X, y, 332)
    # gpr(X_train, X_test, y_train, y_test, X, y, 333)
    # gbrt(X_train, X_test, y_train, y_test, X, y, 334)
    # svr(X_train, X_test, y_train, y_test, X, y, 335)
    # knn(X_train, X_test, y_train, y_test, X, y, 336)
    # ada(X_train, X_test, y_train, y_test, X, y, 337)
    # bag_r(X_train, X_test, y_train, y_test, X, y, 338)
    # tree_extra(X_train, X_test, y_train, y_test, X, y, 339)
    #
    # plt.savefig('scikit_'+fp+'.png', dpi=300)
