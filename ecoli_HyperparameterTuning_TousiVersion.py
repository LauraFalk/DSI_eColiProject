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
import winsound

############################# File Processing ############################# 

ec_csv = pd.read_csv('Data\Processed\ecoli_attributed.csv')

#Drop unecessary columns.
ec_csv = ec_csv.drop(['Unnamed: 0'], axis=1)

# Create the Array
ec_array = np.array(ec_csv)

# Create the Array for each criteria (235 and 575)
ec_x = ec_array[:, 0:-2]
ec_235 = ec_array[:, -2]
ec_575 = ec_array[:, -1]

# Test to ensure this is working. 
# Row 16 is the first row which has a different category for 235 and 575 as of 10/6/2022.
# Prints all rows rather than just the first ones
pd.set_option('display.max_columns', None)
# Print the comparisons
print(ec_csv.loc[[16]])
print(ec_x[16])
print(ec_235[16])
print(ec_575[16])

# Use the MinMax Scaler to appropriately scale predictor data
scaler_ec_x = MinMaxScaler()
ec_x_scl = scaler_ec_x.fit_transform(ec_x)

############################# Binary Classification LR and RC - written by Tousi et. al ############################# 
def Classifier (x, y, model, P):
    # p is number of iteration of 5 fold CV
    confmax_ts_p_iteration = np.zeros((P, 2, 2))
    TPR_ts_p_iteration = np.zeros(P)
    TNR_ts_p_iteration = np.zeros(P)
    FNR_ts_p_iteration = np.zeros(P)
    FPR_ts_p_iteration = np.zeros(P)
    prfs_array_ts_p_iteration = np.zeros((P, 4, 2))
    ##
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
        ##
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
        kf_cl = StratifiedKFold(n_splits=KN, shuffle=True, random_state=p)
        for train_index, test_index in kf_cl.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ##
            model.fit(X_train, y_train)
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
        ##
        confmax_tr_p_iteration[p] = confmax_tr.mean(axis=0)
        TPR_tr_p_iteration[p] = TPR_tr.mean()
        TNR_tr_p_iteration[p] = TNR_tr.mean()
        FPR_tr_p_iteration[p] = FPR_tr.mean()
        FNR_tr_p_iteration[p] = FNR_tr.mean()
        prfs_array_tr_p_iteration[p] = prfs_array_tr.mean(axis=0)
        # #
        confmax_ts_p_iteration[p] = confmax_ts.mean(axis=0)
        TPR_ts_p_iteration[p] = TPR_ts.mean()
        TNR_ts_p_iteration[p] = TNR_ts.mean()
        FPR_ts_p_iteration[p] = FPR_ts.mean()
        FNR_ts_p_iteration[p] = FNR_ts.mean()
        prfs_array_ts_p_iteration[p] = prfs_array_ts.mean(axis=0)
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
    TPR_test = np.mean(TPR_ts_p_iteration, axis=0)
    TNR_test = np.mean(TNR_ts_p_iteration, axis=0)
    FNR_test = np.mean(FNR_ts_p_iteration, axis=0)
    FPR_test = np.mean(FPR_ts_p_iteration, axis=0)
    TPR_train = np.mean(TPR_tr_p_iteration, axis=0)
    TNR_train = np.mean(TNR_tr_p_iteration, axis=0)
    FNR_train = np.mean(FNR_tr_p_iteration, axis=0)
    FPR_train = np.mean(FPR_tr_p_iteration, axis=0)
    

    ##
    return TPR_test, TNR_test, FNR_test, FPR_test, TPR_train, TNR_train, FNR_train, FPR_train
    # return


############################# Classification KLR and KRC - written by Tousi et. al ############################# 

def KClassifier (x, y, model, P): #### Note - Only works for binary classification
    # p is number of iteration of 5 fold CV
    confmax_ts_p_iteration = np.zeros((P, 2, 2))
    TPR_ts_p_iteration = np.zeros(P)
    TNR_ts_p_iteration = np.zeros(P)
    FNR_ts_p_iteration = np.zeros(P)
    FPR_ts_p_iteration = np.zeros(P)
    prfs_array_ts_p_iteration = np.zeros((P, 4, 2))
    ##
    confmax_tr_p_iteration = np.zeros((P, 2, 2))
    TPR_tr_p_iteration = np.zeros(P)
    TNR_tr_p_iteration = np.zeros(P)
    FNR_tr_p_iteration = np.zeros(P)
    FPR_tr_p_iteration = np.zeros(P)
    prfs_array_tr_p_iteration = np.zeros((P, 4, 2))
    for p in range(P):
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
        ##
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
        kf_cl = StratifiedKFold(n_splits=KN, shuffle=True, random_state=p)
        for train_index, test_index in kf_cl.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #Turn on for poly
            poly_trs = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False) 
            X_train = poly_trs.fit_transform(X_train)
            X_test = poly_trs.fit_transform(X_test)
            # Turn on for poly
            ##
            model.fit(X_train, y_train)
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
        confmax_tr_p_iteration[p] = confmax_tr.mean(axis=0)
        TPR_tr_p_iteration[p] = TPR_tr.mean()
        TNR_tr_p_iteration[p] = TNR_tr.mean()
        FPR_tr_p_iteration[p] = FPR_tr.mean()
        FNR_tr_p_iteration[p] = FNR_tr.mean()
        prfs_array_tr_p_iteration[p] = prfs_array_tr.mean(axis=0)
        # #
        confmax_ts_p_iteration[p] = confmax_ts.mean(axis=0)
        TPR_ts_p_iteration[p] = TPR_ts.mean()
        TNR_ts_p_iteration[p] = TNR_ts.mean()
        FPR_ts_p_iteration[p] = FPR_ts.mean()
        FNR_ts_p_iteration[p] = FNR_ts.mean()
        prfs_array_ts_p_iteration[p] = prfs_array_ts.mean(axis=0)
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
    TPR_test = np.mean(TPR_ts_p_iteration, axis=0)
    TNR_test = np.mean(TNR_ts_p_iteration, axis=0)
    FNR_test = np.mean(FNR_ts_p_iteration, axis=0)
    FPR_test = np.mean(FPR_ts_p_iteration, axis=0)
    ##
    TPR_train = np.mean(TPR_tr_p_iteration, axis=0)
    TNR_train = np.mean(TNR_tr_p_iteration, axis=0)
    FNR_train = np.mean(FNR_tr_p_iteration, axis=0)
    FPR_train = np.mean(FPR_tr_p_iteration, axis=0)
    ##
    return TPR_test, TNR_test, FNR_test, FPR_test, TPR_train, TNR_train, FNR_train, FPR_train
    # return



############################# Hyperparameter Tunning for Log regression ############################# 
d_w = 1
w_uplim = 3
w_lolim = 0.001

d_C = 0.01
C_uplim = 1
C_lolim = 0.001


Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((w_uplim-w_lolim)/d_w)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))

for j in range(mt.ceil((w_uplim-w_lolim)/d_w)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        C = C_lolim + i*d_C
        w = w_lolim + j*d_w
        weg = {0: 1, 1: w} ## Note for L_126, weg = {0: 1, 1: w}. For L_1 weg = {0: w, 1: 1}
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = Classifier(ec_x_scl,ec_235, LogisticRegression(C=C, class_weight=weg),1)
        Test_and_train_opt_loop_metric[m, -2] = C
        Test_and_train_opt_loop_metric[m, -1] = w

col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 
'TPR_train', 'TNR_train', 'FnR_train', 'FpR_train', 
'param_1', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Test_and_train_opt_loop_metric_LR_LMP_235_testymctestface3.xlsx')
os.startfile('Test_and_train_opt_loop_metric_LR_LMP_235_testymctestface3.xlsx')

#############################  Hyperparameter Tunning Ridge regression   ############################# 
### THIS OE WORKED!!!
d_w = 1
w_uplim = 100
w_lolim = 0

d_C = 0.1
C_uplim = 6
C_lolim = 4


Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((w_uplim-w_lolim)/d_w)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))

for j in range(mt.ceil((w_uplim-w_lolim)/d_w)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        Alpha = C_lolim + i*d_C
        w = w_lolim + j*d_w
        weg = {0: 1, 1: w} ## Note for L_126, weg = {0: 1, 1: w}. For L_1 weg = {0: w, 1: 1}
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = Classifier(ec_x_scl,ec_235, RidgeClassifier(alpha=Alpha, class_weight=weg),1)
        Test_and_train_opt_loop_metric[m, -2] = Alpha
        Test_and_train_opt_loop_metric[m, -1] = w

col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 
'TPR_train', 'TNR_train', 'FnR_train', 'FpR_train', 
'param_1', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Test_and_train_opt_loop_metric_RC_LMP_235_test.xlsx')
os.startfile('Test_and_train_opt_loop_metric_RC_LMP_235_test.xlsx')

############################# Hyperparameter Tunning Kernel Logistic regression   ############################# 

d_w = 0.1
w_uplim = 5.001
w_lolim = 0.001

d_C = 0.1
C_uplim = 3.001
C_lolim = 0.001


Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((w_uplim-w_lolim)/d_w)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))

for j in range(mt.ceil((w_uplim-w_lolim)/d_w)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        Alpha = C_lolim + i*d_C
        w = w_lolim + j*d_w
        weg = {0: 1, 1: w} ## Note for L_126, weg = {0: 1, 1: w}. For L_1 weg = {0: w, 1: 1}
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = ClassifierK(ec_x_scl,ec_235, LogisticRegression(ca=Alpha, class_weight=weg),1)
        Test_and_train_opt_loop_metric[m, -2] = Alpha
        Test_and_train_opt_loop_metric[m, -1] = w

col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 
'TPR_train', 'TNR_train', 'FnR_train', 'FpR_train', 
'param_1', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Test_and_train_opt_loop_metric_KRC_LMP_235.xlsx')
os.startfile('Test_and_train_opt_loop_metric_KRC_LMP_235.xlsx')


############################# Hyperparameter Tunning Kernel Ridge regression   ############################# 

d_w = 0.1
w_uplim = 5.001
w_lolim = 0.001

d_C = 0.1
C_uplim = 3.001
C_lolim = 0.001


Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((w_uplim-w_lolim)/d_w)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))

for j in range(mt.ceil((w_uplim-w_lolim)/d_w)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        Alpha = C_lolim + i*d_C
        w = w_lolim + j*d_w
        weg = {0: 1, 1: w} ## Note for L_126, weg = {0: 1, 1: w}. For L_1 weg = {0: w, 1: 1}
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = ClassifierK(ec_x_scl,ec_235, RidgeClassifier(alpha=Alpha, class_weight=weg),1)
        Test_and_train_opt_loop_metric[m, -2] = Alpha
        Test_and_train_opt_loop_metric[m, -1] = w

col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 
'TPR_train', 'TNR_train', 'FnR_train', 'FpR_train', 
'param_1', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Test_and_train_opt_loop_metric_KRC_LMP_235.xlsx')
os.startfile('Test_and_train_opt_loop_metric_KRC_LMP_235.xlsx')


#############################  Hyperparameter Tunning for SVM ############################# 
import warnings

warnings.filterwarnings('ignore')
d_C = 0.1
C_uplim = 100
C_lolim = 90.001
#
d_gamm = 0.1
gamm_uplim = 39
gamm_lolim = 35.001
#
Test_and_train_opt_loop_metric = np.zeros(((mt.ceil((gamm_uplim-gamm_lolim)/d_gamm)*mt.ceil((C_uplim-C_lolim)/d_C)), 10))
#
for j in range(mt.ceil((gamm_uplim-gamm_lolim)/d_gamm)):
    for i in range(mt.ceil((C_uplim-C_lolim)/d_C)):
        C = C_lolim + i*d_C
        gamm = gamm_lolim + j*d_gamm
        m = i + j *(mt.ceil((C_uplim-C_lolim)/d_C))
        Test_and_train_opt_loop_metric[m, 0:8] = Classifier(ec_x_scl,ec_235, SVC(kernel='rbf', C=C, gamma=gamm, class_weight='balanced'),1)
        Test_and_train_opt_loop_metric[m, -2] = C
        Test_and_train_opt_loop_metric[m, -1] = gamm
#
        print('C:', C)
    print('gamm:', gamm)
#
#
#
col_names = ['TPR_test','TNR_test','FNR_test','FPR_test', 'TPR_train', 'TNR_train', 'FPR_train', 'FNR_train', 'param_3', 'param2']
pdd = pd.DataFrame(Test_and_train_opt_loop_metric, columns=col_names).to_excel('Test_and_train_opt_loop_metric_SVM_LMP_235_test2.xlsx')
os.startfile('Test_and_train_opt_loop_metric_SVM_LMP_235_test2.xlsx')
# #

############################# Add sound ############################# 
#LP Add sound so I know when to pay attention
winsound.Beep(340, 200)


#######################Work in Progress


    
###############LMP Addition
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)
score = logisticRegr.score(X_test, y_test)
print('LR SCORE: ',score)
    
RidgeClass = RidgeClassifier()
RidgeClass.fit(X_train, y_train)
predictions = RidgeClass.predict(X_test)
score = RidgeClass.score(X_test, y_test)
print('RC SCORE: ',score)        
#############End LMP Addition

