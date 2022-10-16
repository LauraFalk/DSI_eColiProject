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
## Added by LP
from sklearn import metrics
## Read in the data
ec_csv = pd.read_csv('Data\Processed\ecoli_attributed.csv')

#Drop unecessary columns.
ec_csv = ec_csv.drop(['Unnamed: 0'], axis=1)

# Create the Array
ec_array = np.array(ec_csv)

# Create the Array for each criteria (236 and 575)
ec_x = ec_array[:, 0:-2]
ec_236 = ec_array[:, -2]
ec_575 = ec_array[:, -1]

# Test to ensure this is working. 
# Row 16 is the first row which has a different category for 236 and 575 as of 10/6/2022.
# Prints all rows rather than just the first ones
pd.set_option('display.max_columns', None)
# Print the comparisons
print(ec_csv.loc[[16]])
print(ec_x[16])
print(ec_236[16])
print(ec_575[16])

# Use the MinMax Scaler to appropriately scale predictor data
scaler_ec_x = MinMaxScaler()
ec_x_scl = scaler_ec_x.fit_transform(ec_x)


# Split the dataset into test and train. I want to do a 70/30 split.
x_train_236, x_test_236, y_train_236, y_test_236 = train_test_split(
  ec_x_scl, ec_236, test_size=0.3, random_state=24)

x_train_575, x_test_575, y_train_575, y_test_575 = train_test_split(
  ec_x_scl, ec_575_scl, test_size=0.3, random_state=24)

############################################

# Test LR on dataset. Modified from https://towardsdatascience.com
# Make an instance
logisticRegr = LogisticRegression()

# Fit the model using all defaults.
logisticRegr.fit(x_train_236, y_train_236)

# Predict
predictions = logisticRegr.predict(x_test_236)

# Use score method to get accuracy of model
score = logisticRegr.score(x_test_236, y_test_236)
print(score)

###############################
# Test SVC on dataset. Modified from https://www.datacamp.com
clf = svm.SVC(kernel='rbf') # Tousi indicates this may be ideal


ytrainboo = y_train_236.astype(int)
#Train the model using the training sets
clf.fit(x_train_236, y_train_236)

#Predict the response for test dataset
y_pred = clf.predict(x_test_236)

ytestboo = y_test_236.astype(int)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(x_test_236, ytestboo))

################################
# Laura - think about Tmin. It is highly correlated with previous30, 
# Default LR including Tmin is 82.5% success
# Default LR discluding Tmin is 81.4% success.

# Laura - this is a binary 

# Laura - Your variable name is 235 not 236 - reverify which one I'm supposed to use.

