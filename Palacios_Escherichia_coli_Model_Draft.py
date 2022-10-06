# Model is modified from the original script from
#Duan, Jennifer Guohong; Tousi, Erfan; Gundy, Patricia M.; Bright, Kelly; Gerba, Charles P. (2022).
#Data and Code for evaluation of E. Coli in sediment for assessing irrigation water quality using machine learning.
#University of Arizona Research Data Repository.
#Software. https://doi.org/10.25422/azu.data.21096184.

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

## Read in the data
ecoli_csv = pd.read_csv('Data\Processed\ecoli_attributed.csv')

#Drop unecessary columns.
ecoli_csv = ecoli_csv.drop(['Unnamed: 0'], axis=1)

# Create the Array
ecoli_array = np.array(ecoli_csv)

# Create the Array for each criteria (236 and 575)
ecoli_x = ecoli_array[:, 0:-2]
ecoli_236 = ecoli_test[:, -2]
ecoli_575 = ecoli_test[:, -1]

#Test to ensure this is working
pd.set_option('display.max_columns', None)
print(ecoli_csv.loc[[11]])
print(ecoli_x[11])
print(ecoli_236[11])
print(ecoli_575[11])

# LAURA this isn't working, why??
