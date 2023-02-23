import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
import os
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import naive_bayes
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
import sklearn_lvq
from sklearn.linear_model import _logistic
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import NuSVC
from sklearn import discriminant_analysis
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn import tree
import math as mt
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import scoring
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import precision_score, recall_score, roc_curve

# LP addition
import time
time.sleep(10)

inp = pd.read_excel(os.path.join(os.getcwd(),'Tousi_Replication/Input_76_data point.xlsx'))
input_data = pd.DataFrame(inp)
input_data.drop(input_data.iloc[:, 0:4], axis=1, inplace=True)
input_data.rename(columns=input_data.iloc[0], inplace=True)
input_data.drop([0], inplace=True)
#######################################
#considering E. coli
# below means drop all the ements of (:) of colum 12 and 14
in_E = input_data.drop(input_data.iloc[:, [14, 11]], axis=1)# in_E_np = np.array(input_data)
in_E_np = np.array(in_E)

############### Total coliform TC
# in_TC = input_data.drop(input_data.iloc[:, [15, 12]], axis=1)
# in_TC_np = np.array(in_TC)
# # print(in_TC_np)

########################################
X_Ec = in_E_np[:, 0:-1]
y_Ec = in_E_np[:, -1]

# X_TC = in_TC_np[:, 0:-1]
# y_TC = in_TC_np[:, -1]

X_Ec_pure = X_Ec


###############  E. coli  ##############################
##### Discarding EC in sed and D50 and non-dimensional shear stress from feathers as it is not readily measurable
X_Ec = np.delete(X_Ec, [10, 11, 12], axis=1)

#####         E. coli >1        ######## feature reduction for Coliniarity and unrelevanet featers #############

X_Ec_1 = np.delete(X_Ec, [0, 4, 5, 6, 8], axis=1)

#Removed this, is not used in output - LMP
#X_Ec_1_snc = np.delete(X_Ec_pure, [0, 4, 5, 6, 8, 9], axis=1) ##### snc = means with sediment data for noncampring having 7 features of  RH, Air temp, Turb τ*, S_Ec, D50, pH

X_Ec_1_sc = np.delete(X_Ec_pure, [0, 4, 5, 6, 8, 9, 1, 12], axis=1) ##### sc = means with sediment data for campring having 5 features of  RH, Air temp, Turb τ*, S_Ec
# print(X_Ec_1_sc)
##########     E. coli >126      ######## feature reduction for Coliniarity and unrelevanet featers #############
X_Ec_126 = np.delete(X_Ec, [3, 5, 6, 7], axis=1)

X_Ec_126_5f = np.delete(X_Ec_126, [1], axis=1)

# print(X_Ec_126_5f)

# Removed this, is not used in output - LMP
# X_Ec_126_snc = np.delete(X_Ec_pure, [3, 5, 6, 7, 10, 12], axis=1)

X_Ec_126_sc = np.delete(X_Ec_pure, [3, 5, 6, 7, 10, 12, 9, ], axis=1)

#Removed this, is not used in output - LMP
# X_Ec_126_sc_5f = np.delete(X_Ec_126_sc, [], axis=1)
# #
# print(X_Ec_126_sc_5f)
############### TC  ##############################
# ##### Discarding TC in sed and D50 and non-dimensional shear stress from feathers as it is not readily measurable
# X_TC = np.delete(X_TC, [10, 11, 12], axis=1)
#
# ##### COLONIARITY Discarding  Salinity, TDS, and Air temp from feathers as it is not readily measurable
# X_TC = np.delete(X_TC, [5, 6, 7], axis=1)

##################### Labeling E. coli
y_Ec_label_1 = np.where(y_Ec > 1, 1, 0) ########### Positive label is 1, means y_lab = 1

y_Ec_label_126 = np.where(y_Ec > 126, 1, 0) ########### Positive label is 1, means y_lab = 1

# rho_1, pval = stats.spearmanr(y_Ec_label_126,X_Ec)
# pd.DataFrame(rho_1).to_excel('rho_1.xlsx')
# os.startfile('rho_1.xlsx')

################# Forward feature selection

# ##################### rho
# Xy_labeld, pval = stats.spearmanr(y_Ec_label, X_Ec)
# pd.DataFrame(Xy_labeld).to_excel('Xy_labedl_rho.xlsx')
# os.startfile('Xy_labedl_rho.xlsx')

# rho_xy_126, pval = stats.spearmanr(y_Ec_label_126, X_Ec_pure)
# pd.DataFrame(rho_xy_126).to_excel('rho_Xy_126.xlsx')
# os.startfile('rho_Xy_126.xlsx')

################ PCA demonstartion and plot E. coli

# print(y_Ec_label)
# print(y_Ec_label.sum()/y_Ec_label.shape[0]*100)
#
# pca = decomposition.PCA(n_components=3)
# pca_x_Ec = pca.fit_transform(X_Ec)
# # print(pca_x_Ec)
# #
# pca_x_Ec_df = pd.DataFrame(data=pca_x_Ec, columns=['PC1', 'PC2', 'PC3'])
# pca_x_Ec_df['cluster'] = y_Ec_label
# pca_x_Ec_df.head()
# # print(pca_x_Ec_df)
# # pca_x_Ec_df.to_excel('PCA_Ec.xlsx')
# # os.startfile('PCA_Ec.xlsx')
# print(pca.explained_variance_ratio_)
#
# sns.lmplot(x='PC1', y='PC2',  data=pca_x_Ec_df, fit_reg=False, hue='cluster', legend=True, scatter_kws={"s":80})
# plt.show()
# # # #
# # # ######################### 3D plot
# #
# fig = plt.figure(1, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# ax.scatter(pca_x_Ec[:, 0], pca_x_Ec[:, 1], pca_x_Ec[:, 2], c=y_Ec_label)
# plt.show()

############ remove outliner that is in row 31 starting from 0 as python style
# X_Ec_mod = np.delete(X_Ec, [31, 32], axis=0)
# y_Ec_mod = np.delete(y_Ec, [31, 32], axis=0)
#
# pca_mod = decomposition.PCA(n_components=3)
# pca_x_Ec_mod = pca_mod.fit_transform(X_Ec_mod)
# print(pca_mod.explained_variance_ratio_)
#
# y_Ec_mod_label = np.where(y_Ec_mod > 1, 1, 0)
#
# ######################### 3D plot
# # fig = plt.figure(1, figsize=(4, 3))
# # plt.clf()
# # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# # ax.scatter(pca_x_Ec_mod[:, 0], pca_x_Ec_mod[:, 1], pca_x_Ec_mod[:, 2], c=y_Ec_mod_label)
# # plt.show()
#
# ############# 2D PCA plot
#
# pca_x_Ec_mod_df = pd.DataFrame(data=pca_x_Ec_mod, columns=['PC1', 'PC2', 'PC3'])
# pca_x_Ec_mod_df['cluster'] = y_Ec_mod_label
# pca_x_Ec_mod_df.head()
# plt.show()

#######################################
scalerX_Ec_1 = MinMaxScaler()
X_Ec_scl_1 = scalerX_Ec_1.fit_transform(X_Ec_1)

# Removed this, is not used in output - LMP
# scalerX_Ec_1_snc = MinMaxScaler()
# X_Ec_scl_1_snc = scalerX_Ec_1_snc.fit_transform(X_Ec_1_snc)

scalerX_Ec_1_sc = MinMaxScaler()
X_Ec_scl_1_sc = scalerX_Ec_1_sc.fit_transform(X_Ec_1_sc)


scalerX_Ec_126 = MinMaxScaler()
X_Ec_scl_126 = scalerX_Ec_126.fit_transform(X_Ec_126)

# Removed this, is not used in output - LMP
# scalerX_Ec_126_5f = MinMaxScaler()
# X_Ec_scl_126_5f = scalerX_Ec_126_5f.fit_transform(X_Ec_126_5f)

# Removed this, is not used in output - LMP
# scalerX_Ec_126_snc = MinMaxScaler()
# X_Ec_scl_126_snc = scalerX_Ec_126_snc.fit_transform(X_Ec_126_snc)

scalerX_Ec_126_sc = MinMaxScaler()
X_Ec_scl_126_sc = scalerX_Ec_126_sc.fit_transform(X_Ec_126_sc)

# Removed this, is not used in output - LMP
# scalerX_Ec_126_sc_5f = MinMaxScaler()
# X_Ec_scl_126_sc_5f = scalerX_Ec_126_sc_5f.fit_transform(X_Ec_126_sc_5f)

################################################ THIS CODE need to be revised not verficed
# def backward_feature_selection(XX, YY, model, feature_labels):
#     least_to_most_important_feature = []
#     for k in range(XX.shape[1]-1):
#         if k == 0:
#             True
#         else:
#             print('number of already removed feather:', k)
#             XX = np.delete(XX, [least_important_feature_index], axis=1)
#             # print(XX)
#         tot_num_feat = XX.shape[1]
#         tr_f1_feth = np.zeros(tot_num_feat+1)
#         ts_f1_feth = np.zeros(tot_num_feat+1)
#         # tr_scr_feth = np.zeros(tot_num_feat+1)
#         # ts_scr_feth = np.zeros(tot_num_feat+1)
#         for i in range(tot_num_feat+1):
#             print('right after start of loop', i)
#             KN = 5
#             n = 0
#             tr_f1 = np.zeros(KN)
#             ts_f1 = np.zeros(KN)
#             # tr_mse = np.zeros(KN)
#             # ts_mse = np.zeros(KN)
#             # tr_scr = np.zeros(KN)
#             # ts_scr = np.zeros(KN)
#             kf = StratifiedKFold(n_splits=KN, shuffle=True, random_state=1)
#             if i == 0:
#                 for train_index, test_index in kf.split(XX, YY):
#                     # print("TRAIN:", train_index, "TEST:", test_index)
#                     X_train, X_test = XX[train_index], XX[test_index]
#                     y_train, y_test = YY[train_index], YY[test_index]
#                     model.fit(X_train, y_train)
#                     tr_f1[n] = f1_score(y_train, model.predict(X_train), average='weighted')
#                     ts_f1[n] = f1_score(y_test, model.predict(X_test), average='weighted')
#                     print(ts_f1)
#                     # tr_scr[n] = model.score(X_train, y_train)
#                     # ts_scr[n] = model.score(X_test, y_test)
#                     n += 1
#
#             else:
#                 XXX = np.delete(XX, [i-1], axis=1)
#                 for train_index, test_index in kf.split(XXX, YY):
#
#                     # print(np.delete(XX, [i-1], axis=1))
#                     # print('erfan')
#                     # print("TRAIN:", train_index, "TEST:", test_index)
#                     X_train, X_test = XXX[train_index], XXX[test_index]
#                     y_train, y_test = YY[train_index], YY[test_index]
#                     model.fit(X_train, y_train)
#                     # tr_mse[n] = mean_squared_error(y_train, model.predict(X_train))
#                     tr_f1[n] = f1_score(y_train, model.predict(X_train), average='weighted')
#                     ts_f1[n] = f1_score(y_test, model.predict(X_test), average='weighted')
#                     print(ts_f1)
#                     # tr_scr[n] = model.score(X_train, y_train)
#                     # ts_scr[n] = model.score(X_test, y_test)
#                     n += 1
#
#             tr_f1_feth[i] = tr_f1.mean()
#             ts_f1_feth[i] = ts_f1.mean()
#             print(ts_f1_feth)
#
#             # tr_scr_feth[i] = tr_scr.mean()
#             # ts_scr_feth[i] = ts_scr.mean()
#         print('ts_f1: ', ts_f1_feth)
#         print('tr_f1: ', tr_f1_feth)
#
#         # tr_mse_select_feat = tr_mse_feth
#         # tr_mse_select_feat_change_from_base_of_iteraltion = abs((tr_mse_feth-tr_mse_feth[0])/tr_mse_feth[0])*100
#         # j = np.where(tr_mse_select_feat_change_from_base_of_iteraltion[1::] == min(tr_mse_select_feat_change_from_base_of_iteraltion[1::]))
#
#         j = np.where(ts_f1_feth[1::] == min(ts_f1_feth))
#         print(j)
#         least_important_feature_index = int(np.array(j))
#         print(least_important_feature_index)
#
#         least_to_most_important_feature.append(feature_labels[least_important_feature_index])
#         feature_labels = np.delete(feature_labels, least_important_feature_index)
#
#         print(least_to_most_important_feature)
#
#
#     least_to_most_important_feature.append(feature_labels[0])
#
#
#     # return
#
#     return print('featuress_based_on_importance_from_lest_the_most_important: ', least_to_most_important_feature), pd.DataFrame(least_to_most_important_feature).to_excel('least_to_most_important_feature.xlsx'), os.startfile('least_to_most_important_feature.xlsx')


# backward_feature_selection(X_Ec_126_sc, y_Ec_label_126, SVC(kernel='rbf', class_weight='balanced'), ['Wtenp','pH', 'Turb.', 'Conduct', 'Depth', 'SE_coli'])

# mod = SVC(kernel='rbf', class_weight='balanced')
# mod.fit(X_Ec_scl_126_sc, y_Ec_label_126)
# print(f1_score(y_Ec_label_126, mod.predict(X_Ec_scl_126_sc), average='weighted'))
# print(precision_recall_fscore_support(y_Ec_label_126, mod.predict(X_Ec_scl_126_sc)))
############### Feature selection
# feat_name = ('Wtenp','pH', 'Turb.', 'Conduct', 'Depth', 'SE_coli')
# # feat_name = ('Wtenp','pH', 'Turb.', 'Conduct', 'Depth', 'V')
#
# sfs = SFS(SVC(kernel='rbf', class_weight='balanced'), k_features=5, forward=True, floating=False, scoring='roc_auc', cv=5)
# sfs = sfs.fit(X_Ec_scl_126_sc, y_Ec_label_126, custom_feature_names=feat_name)
# print(sfs.subsets_)


# fig1 = plot_sfs(sfs.get_metric_dict(),
#                 kind='std_dev',
#                 figsize=(6, 4))
# plt.ylim([0.8, 1])
# plt.title('Sequential Forward Selection (w. StdDev)')
# plt.grid()
# plt.show()


##########################################
# TP_ts[n] = confmax_ts[n, 1, 1]  # TP means class 1, E.coli > threshold, y_cls_er = np.where(y1 > 10, 1, 0)
# TN_ts[n] = confmax_ts[n, 0, 0]
# FN_ts[n] = confmax_ts[n, 1, 0]
# FP_ts[n] = confmax_ts[n, 0, 1]
# confmax_tr_p_iteration = confmax_tr
# TNR_tr_p_iteration = TNR_tr.mean()
# FPR_tr_p_iteration = FPR_tr.mean()
# FNR_tr_p_iteration = FNR_tr.mean()
# prfs_array_tr_p_iteration = prfs_array_tr

########################### Clasiificaiton
def Classifier (x, y, model, P): #### Note this Clssider only works for binary classification
    # p is number of iteration of 5 fold CV
    confmax_ts_p_iteration = np.zeros((P, 2, 2))
    TPR_ts_p_iteration = np.zeros(P)
    TNR_ts_p_iteration = np.zeros(P)
    FNR_ts_p_iteration = np.zeros(P)
    FPR_ts_p_iteration = np.zeros(P)
    prfs_array_ts_p_iteration = np.zeros((P, 4, 2))

    confmax_tr_p_iteration = np.zeros((P, 2, 2))
    TPR_tr_p_iteration = np.zeros(P)
    TNR_tr_p_iteration = np.zeros(P)
    FNR_tr_p_iteration = np.zeros(P)
    FPR_tr_p_iteration = np.zeros(P)
    prfs_array_tr_p_iteration = np.zeros((P, 4, 2))
    for p in range(P):
        # print('Loop: ', p)
        KN = 5
        n = 0
        accr_ts = np.zeros(KN)
        prfs_array_ts = np.zeros((KN, 4, 2))  # NOTE: right side numbers are for class 1
        confmax_ts = np.zeros((KN, 2, 2))
        TPR_ts = np.zeros(KN)
        TNR_ts = np.zeros(KN)
        FNR_ts = np.zeros(KN)
        FPR_ts = np.zeros(KN)
        TP_ts = np.zeros(KN)
        TN_ts = np.zeros(KN)
        FP_ts = np.zeros(KN)
        FN_ts = np.zeros(KN)

        accr_tr = np.zeros(KN)
        prfs_array_tr = np.zeros((KN, 4, 2))  # NOTE: right side numbers are for class 1
        confmax_tr = np.zeros((KN, 2, 2))
        TPR_tr = np.zeros(KN)
        TNR_tr = np.zeros(KN)
        FNR_tr = np.zeros(KN)
        FPR_tr = np.zeros(KN)
        TP_tr = np.zeros(KN)
        TN_tr = np.zeros(KN)
        FP_tr = np.zeros(KN)
        FN_tr = np.zeros(KN)
        # b = 0
        kf_cl = StratifiedKFold(n_splits=KN, shuffle=True, random_state=p)
        for train_index, test_index in kf_cl.split(x, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(train_index)
            # print(y_train)
            print(test_index)
            # print(y_test)

            # pd.DataFrame(np.concatenate((train_index,test_index))).to_excel(f'fold:{b}.xlsx')         # ################### Ploynominal Kernel
            # os.startfile(f'fold: {b}.xlsx')
            # b += 1
            # LP - Turn on following three rows for kernels.
            #poly_trs = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
            #X_train = poly_trs.fit_transform(X_train)
            #X_test = poly_trs.fit_transform(X_test)
            
            # # print(X_train.shape)
            ##################### Model fit
            model.fit(X_train, y_train)
            # tree.plot_tree(model)
            # plt.show()



            # Metrics on Test set
            confmax_ts[n] = confusion_matrix(y_test, model.predict(X_test))
            accr_ts[n] = accuracy_score(y_test, model.predict(X_test))
            TP_ts[n] = confmax_ts[n, 1, 1]# TP means class 1, E.coli > threshold, y_cls_er = np.where(y1 > 10, 1, 0)
            TN_ts[n] = confmax_ts[n, 0, 0]
            FN_ts[n] = confmax_ts[n, 1, 0]
            FP_ts[n] = confmax_ts[n, 0, 1]
            TPR_ts[n] = TP_ts[n] /(TP_ts[n]+FN_ts[n]) #TPR is recall
            TNR_ts[n] = TN_ts[n] / (TN_ts[n] + FP_ts[n])
            FNR_ts[n] = FN_ts[n]/(TP_ts[n]+FN_ts[n])
            FPR_ts[n] = FP_ts[n]/(FP_ts[n]+TN_ts[n])
            prfs_array_ts[n, :, :] = precision_recall_fscore_support(y_test, model.predict(X_test))

            # Metrics on Train set
            confmax_tr[n] = confusion_matrix(y_train, model.predict(X_train))
            accr_tr[n] = accuracy_score(y_train, model.predict(X_train))
            TP_tr[n] = confmax_tr[n, 1, 1]# TP means class 1, E.coli > threshold, y_cls_er = np.where(y1 > 10, 1, 0)
            TN_tr[n] = confmax_tr[n, 0, 0]
            FN_tr[n] = confmax_tr[n, 1, 0]
            FP_tr[n] = confmax_tr[n, 0, 1]
            TPR_tr[n] = TP_tr[n] /(TP_tr[n]+FN_tr[n])
            TNR_tr[n] = TN_tr[n] / (TN_tr[n] + FP_tr[n])
            FNR_tr[n] = FN_tr[n]/(TP_tr[n]+FN_tr[n])
            FPR_tr[n] = FP_tr[n]/(FP_tr[n]+TN_tr[n])
            prfs_array_tr[n, :, :] = precision_recall_fscore_support(y_train, model.predict(X_train))
            n += 1

        # #
        # print('train metrics in CV of loop:', p)
        # print('TPR train: ', TPR_tr)
        # print('TNR train: ', TNR_tr)
        # print('FPR train: ', FPR_tr)
        # print('FNR train: ', FNR_tr)
        # print('prfs train: ', prfs_array_tr)
        confmax_tr_p_iteration[p] = confmax_tr.mean(axis=0)
        TPR_tr_p_iteration[p] = TPR_tr.mean()
        TNR_tr_p_iteration[p] = TNR_tr.mean()
        FPR_tr_p_iteration[p] = FPR_tr.mean()
        FNR_tr_p_iteration[p] = FNR_tr.mean()
        prfs_array_tr_p_iteration[p] = prfs_array_tr.mean(axis=0)
        # #
        # print('test metrics in CV of loop:', p)
        # print('TPR test: ', TPR_ts)
        # print('TNR test: ', TNR_ts)
        # print('FPR test: ', FPR_ts)
        # print('FNR test: ', FNR_ts)
        # print('prfs test: ', prfs_array_ts)
        confmax_ts_p_iteration[p] = confmax_ts.mean(axis=0)
        TPR_ts_p_iteration[p] = TPR_ts.mean()
        TNR_ts_p_iteration[p] = TNR_ts.mean()
        FPR_ts_p_iteration[p] = FPR_ts.mean()
        FNR_ts_p_iteration[p] = FNR_ts.mean()
        prfs_array_ts_p_iteration[p] = prfs_array_ts.mean(axis=0)


        # print(accr_scr)
        # print( 'mean precsision = ', press.mean())
        # print( 'mean recal = ', recal.mean())
        # print( 'mean f1_scr = ', f1_scr.mean())

        # print(classification_report(y_test_list, predic_y_test_list))
        # print(precision_recall_fscore_support(y_test_list, predic_y_test_list))
        # prfs = np.array(precision_recall_fscore_support(y_test_list, predic_y_test_list))
        # print(prfs_array.shape)

    print('#### train results of all  p iteration ###############################################')
    print('mean of TPR train: ', np.mean(TPR_tr_p_iteration, axis=0))
    print('mean of TNR train: ', np.mean(TNR_tr_p_iteration, axis=0))
    print('mean of FNR train: ', np.mean(FNR_tr_p_iteration, axis=0))
    print('mean of FPR train: ', np.mean(FPR_tr_p_iteration, axis=0))
    # print('mean accuracy train: ', np.mean(accr_tr))
    print('Train left clmn is cls 0, right clmn cls1, '
          'rows are pression, recal, f1-score, support', np.mean(prfs_array_tr_p_iteration, axis=0))

    print('#### test results of all  p iteration ###############################################')
    print('mean of TPR test: ', np.mean(TPR_ts_p_iteration, axis=0))
    print('mean of TNR test: ', np.mean(TNR_ts_p_iteration, axis=0))
    print('mean of FNR test: ', np.mean(FNR_ts_p_iteration, axis=0))
    print('mean of FPR test: ', np.mean(FPR_ts_p_iteration, axis=0))
    # print('mean accuracy: ', np.mean(accr_ts))
    print('left clmn is cls 0, right clmn cls1, '
          'rows are pression, recal, f1-score, support', np.mean(prfs_array_ts_p_iteration, axis=0))
    TPR_test = np.mean(TPR_ts_p_iteration, axis=0)
    TNR_test = np.mean(TNR_ts_p_iteration, axis=0)
    FNR_test = np.mean(FNR_ts_p_iteration, axis=0)
    FPR_test = np.mean(FPR_ts_p_iteration, axis=0)
    #
    #
    TPR_train = np.mean(TPR_tr_p_iteration, axis=0)
    TNR_train = np.mean(TNR_tr_p_iteration, axis=0)
    FNR_train = np.mean(FNR_tr_p_iteration, axis=0)
    FPR_train = np.mean(FPR_tr_p_iteration, axis=0)

    return TPR_test, TNR_test, FNR_test, FPR_test, TPR_train, TNR_train, FNR_train, FPR_train
    # return


# wegt = {0:1, 1:12.7}
# Classifier(X_Ec_scl_126_sc, y_Ec_label_126, RidgeClassifier(alpha=0.501, class_weight=wegt), 1)




################# Hyperparameter Tunning for SVM #############
d_C = 0.1
C_uplim = 5
C_lolim = 0.001
#
d_gamm = 0.1
gamm_uplim = 8
gamm_lolim = 0.001
#
Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((gamm_uplim-gamm_lolim)/d_gamm)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))
#
for j in range(mt.ceil((gamm_uplim-gamm_lolim)/d_gamm)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        C = C_lolim + i*d_C
        gamm = gamm_lolim + j*d_gamm
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = Classifier(X_Ec_scl_126, y_Ec_label_126, SVC(kernel='rbf', C=C, gamma=gamm, class_weight='balanced'),1)
        Test_and_train_opt_loop_metric[m, -2] = C
        Test_and_train_opt_loop_metric[m, -1] = gamm
#
        print('C:', C)
    print('gamm:', gamm)
#
#
#
col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 'TPR_train', 'TNR_train', 'FPR_train', 'FNR_train', 'param_3', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Tousi_Replication\\Tuning_Outputs\\Test_and_train_opt_loop_metric_SVM_FES_126.xlsx')
os.startfile('Tousi_Replication\\Tuning_Outputs\\Test_and_train_opt_loop_metric_SVM_FES_126.xlsx')
# #

import winsound
duration = 200  # milliseconds
freq = 340  # Hz
winsound.Beep(freq, duration)

################## Hyperparameter Tunning for Log and Ridge regression   #############

d_C = 0.1
C_uplim = 8
C_lolim = 0.001

d_w = 0.1
w_uplim = 17
w_lolim = 5

Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((w_uplim-w_lolim)/d_w)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))

for j in range(mt.ceil((w_uplim-w_lolim)/d_w)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        C = C_lolim + i*d_C
        w = w_lolim + j*d_w
        weg = {0: 1, 1: w} ## Note for L_126, weg = {0: 1, 1: w}. For L_1 weg = {0: w, 1: 1}
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = Classifier(X_Ec_scl_126,y_Ec_label_126, RidgeClassifier(alpha=C, class_weight=weg),1)
        Test_and_train_opt_loop_metric[m, -2] = C
        Test_and_train_opt_loop_metric[m, -1] = w

        # print('C:', C)
    # print('w:', w)

col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 'TPR_train', 'TNR_train', 'FPR_train', 'FNR_train', 'param_3', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Tousi_Replication\\Tuning_Outputs\\Test_and_train_opt_loop_metric_LR_FES_126.xlsx')
os.startfile('Tousi_Replication\\Tuning_Outputs\\Test_and_train_opt_loop_metric_LR_FES_126.xlsx')
# 

# Added this so I know when to pay attention
import winsound
duration = 200  # milliseconds
freq = 340  # Hz
winsound.Beep(freq, duration)

################################ Final models after all CV on hyperparameter tunning_ Final model is trained over all the data set.
############# SVM with kerenl rbf for both levels and both feature sets of having sed and excluding sed data

##################### E.coli > 1

########## Final tunned models
# Classifier(X_Ec_scl_126_sc, y_Ec_label_126, SVC(kernel='rbf', class_weight='balanced', C=0.601, gamma=5.301), 1)
# Classifier(X_Ec_scl_126, y_Ec_label_126, SVC(kernel='rbf', class_weight='balanced', C=0.301, gamma=5.501), 1)

wegt = {0:2.5, 1:1}
Classifier(X_Ec_scl_1_sc, y_Ec_label_1, RidgeClassifier(alpha=0.801, class_weight=wegt), 1)




