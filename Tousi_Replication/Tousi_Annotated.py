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
#Altered this from "logistic" because error message.
from sklearn.linear_model import LogisticRegression
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



##LP Read in the data
inp = pd.read_excel('Input_76_data point.xlsx')
input_data = pd.DataFrame(inp)

##LP Drop the null columns (excludes private locational data.)
input_data.drop(input_data.iloc[:, 0:4], axis=1, inplace=True)

##LP populates column headers
input_data.rename(columns=input_data.iloc[0], inplace=True)

##LP drops the row which contained column headers from original file.
input_data.drop([0], inplace=True)


#######################################
#considering E. coli
# below means drop all the ements of (:) of colum 12 and 14
##LP Drop TOC columns, only predictors and e. coli remain.
in_E = input_data.drop(input_data.iloc[:, [14, 11]], axis=1)# in_E_np = np.array(input_data)

##LP Create the array
in_E_np = np.array(in_E)

############### Total coliform TC
## This was taken out likely due to TOC not being predictable enough?

# in_TC = input_data.drop(input_data.iloc[:, [15, 12]], axis=1)
# in_TC_np = np.array(in_TC)
# # print(in_TC_np)

########################################
##LP X Array
X_Ec = in_E_np[:, 0:-1]
##LP Y array
y_Ec = in_E_np[:, -1]

# X_TC = in_TC_np[:, 0:-1]
# y_TC = in_TC_np[:, -1]

X_Ec_pure = X_Ec


###############  E. coli  ##############################
##### Discarding EC in sed and D50 and non-dimensional shear stress from feathers as it is not readily measurable
##LP - discards some variables from the array
X_Ec = np.delete(X_Ec, [10, 11, 12], axis=1)

#####         E. coli >1        ######## feature reduction for Coliniarity and unrelevanet featers #############
##LP - discards some variables from the array
X_Ec_1 = np.delete(X_Ec, [0, 4, 5, 6, 8], axis=1)

X_Ec_1_snc = np.delete(X_Ec_pure, [0, 4, 5, 6, 8, 9], axis=1) ##### snc = means with sediment data for noncampring having 7 features of  RH, Air temp, Turb τ*, S_Ec, D50, pH

X_Ec_1_sc = np.delete(X_Ec_pure, [0, 4, 5, 6, 8, 9, 1, 12], axis=1) ##### sc = means with sediment data for campring having 5 features of  RH, Air temp, Turb τ*, S_Ec
# print(X_Ec_1_sc)
##########     E. coli >126      ######## feature reduction for Coliniarity and unrelevanet featers #############
##LP - discards some variables from the array
X_Ec_126 = np.delete(X_Ec, [3, 5, 6, 7], axis=1)

X_Ec_126_5f = np.delete(X_Ec_126, [1], axis=1)

#print(X_Ec_126_5f)

X_Ec_126_snc = np.delete(X_Ec_pure, [3, 5, 6, 7, 10, 12], axis=1)

X_Ec_126_sc = np.delete(X_Ec_pure, [3, 5, 6, 7, 10, 12, 9, ], axis=1)

X_Ec_126_sc_5f = np.delete(X_Ec_126_sc, [], axis=1)
FES_X_Ec_126_sc_5f = np.delete(X_Ec_126_sc, [5], axis=1)
 
##################### Labeling E. coli
y_Ec_label_1 = np.where(y_Ec > 1, 1, 0) ########### Positive label is 1, means y_lab = 1

y_Ec_label_126 = np.where(y_Ec > 126, 1, 0) ########### Positive label is 1, means y_lab = 1

#######################################
##LP Automatic scaling of variables
scalerX_Ec_1 = MinMaxScaler()
X_Ec_scl_1 = scalerX_Ec_1.fit_transform(X_Ec_1)

scalerX_Ec_1_snc = MinMaxScaler()
X_Ec_scl_1_snc = scalerX_Ec_1_snc.fit_transform(X_Ec_1_snc)

scalerX_Ec_1_sc = MinMaxScaler()
X_Ec_scl_1_sc = scalerX_Ec_1_sc.fit_transform(X_Ec_1_sc)


scalerX_Ec_126 = MinMaxScaler()
X_Ec_scl_126 = scalerX_Ec_126.fit_transform(X_Ec_126)

scalerX_Ec_126_5f = MinMaxScaler()
X_Ec_scl_126_5f = scalerX_Ec_126_5f.fit_transform(X_Ec_126_5f)

scalerX_Ec_126_snc = MinMaxScaler()
X_Ec_scl_126_snc = scalerX_Ec_126_snc.fit_transform(X_Ec_126_snc)

scalerX_Ec_126_sc = MinMaxScaler()
X_Ec_scl_126_sc = scalerX_Ec_126_sc.fit_transform(X_Ec_126_sc)

scalerX_Ec_126_sc_5f = MinMaxScaler()
X_Ec_scl_126_sc_5f = scalerX_Ec_126_sc_5f.fit_transform(X_Ec_126_sc_5f)

scalerFES_X_Ec_126_sc_5f = MinMaxScaler()
FES_X_Ec_scl_126_sc_5f = scalerFES_X_Ec_126_sc_5f.fit_transform(FES_X_Ec_126_sc_5f)


########################### Clasiificaiton
##LP Def defines a function. This whole classification loop makes the table 3 output in the paper.
def Classifier (x, y, model, P): #### Note this Clssider only works for binary classification
    # p is number of iteration of 5 fold CV
    confmax_ts_p_iteration = np.zeros((P, 2, 2)) ##LP create an array of zeros (numpy zeros)
    ##LP what is TS and TR?
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
        print('Loop: ', p)
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
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(train_index)
            # print(y_train)
            print(test_index)
             
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

        print('train metrics in CV of loop:', p)
        print('TPR train: ', TPR_tr)
        print('TNR train: ', TNR_tr)
        print('FPR train: ', FPR_tr)
        print('FNR train: ', FNR_tr)
        print('prfs train: ', prfs_array_tr)
        
        confmax_tr_p_iteration[p] = confmax_tr.mean(axis=0)
        TPR_tr_p_iteration[p] = TPR_tr.mean()
        TNR_tr_p_iteration[p] = TNR_tr.mean()
        FPR_tr_p_iteration[p] = FPR_tr.mean()
        FNR_tr_p_iteration[p] = FNR_tr.mean()
        prfs_array_tr_p_iteration[p] = prfs_array_tr.mean(axis=0)
        # #
        print('test metrics in CV of loop:', p)
        print('TPR test: ', TPR_ts)
        print('TNR test: ', TNR_ts)
        print('FPR test: ', FPR_ts)
        print('FNR test: ', FNR_ts)
        print('prfs test: ', prfs_array_ts)
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
    print('Train left clmn is cls 0, right clmn cls1, '
          'rows are pression, recal, f1-score, support', np.mean(prfs_array_tr_p_iteration, axis=0))

    print('#### test results of all  p iteration ###############################################')
    print('mean of TPR test: ', np.mean(TPR_ts_p_iteration, axis=0))
    print('mean of TNR test: ', np.mean(TNR_ts_p_iteration, axis=0))
    print('mean of FNR test: ', np.mean(FNR_ts_p_iteration, axis=0))
    print('mean of FPR test: ', np.mean(FPR_ts_p_iteration, axis=0))
    print('left clmn is cls 0, right clmn cls1, '
          'rows are pression, recal, f1-score, support', np.mean(prfs_array_ts_p_iteration, axis=0))
     
    return

## LP the inputs have more sig figs than the paper included in the explanatory table. 
## LPI want to try and run the logreg grid in order to make this work.
wegt = {0:1, 1:15.7}
Classifier(X_Ec_scl_126_sc_5f, y_Ec_label_126, LogisticRegression(C=11.301, class_weight=wegt), 1)



wegt = {0:1, 1:5.5}
Classifier(FES_X_Ec_scl_126_sc_5f, y_Ec_label_126, SVC(C=0.3, class_weight=wegt), 1)
 





