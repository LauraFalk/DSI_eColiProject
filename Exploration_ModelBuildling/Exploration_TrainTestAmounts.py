import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

############################# File Processing #############################

# ec_csv = pd.read_csv('Data\Processed\ecoli_attributed.csv')
ec_csv = pd.read_csv('C:\\Users\\Laura\\Desktop\\DSI_eColi\\DSI_eColiProject\\Data\\Processed\\ecoli_attributed2.csv')
# Drop unnecessary columns.
ec_csv = ec_csv.drop(['Unnamed: 0'], axis=1)

# Create the Array
ec_array = np.array(ec_csv)

# Create the Array for each criterion (235 and 575)
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

# This will loop through things
rangelist = np.arange(1, 20, 1)
randomlist = np.arange(3, 16, 3)  # 16
finalScore = {}

for i in rangelist:
    # print(i)
    untrainedSum = 0
    gridSum = 0
    TousiSum = 0
    scoreCount = 0
    USVCSum = 0
    GSVCSum = 0
    TSVCSum = 0
    URCSum = 0
    XGBSum = 0
    GXGSum = 0
    ts = (i / 2) * 0.1
    rTest = ts*917

    #print("test Proportion:", ts)
    for j in randomlist:
        x_train, x_test, y_train, y_test = train_test_split(ec_x_scl, ec_575, test_size=ts, random_state=j)

        # Logistic Regression
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        # print('Untrained', i, j, ':', metrics.accuracy_score(y_test, logisticRegr.predict(x_test)))

        # SVC
        USVC = SVC()
        USVC.fit(x_train, y_train)

        URC = RidgeClassifier()
        URC.fit(x_train, y_train)

        # Grid search
        grid_params = {"C": np.arange(1, 10, 0.1), "penalty": ["l1", "l2", "elasticnet"],
                       "solver": ['newton-cg', 'lbfgs', 'liblinear']}

        logreg = LogisticRegression()
        logreg_cv = GridSearchCV(logreg, grid_params, cv=3, verbose=0, return_train_score=True)
        logreg_cv.fit(x_train, y_train)
        # print("tuned hyperparameters :(best parameters) ", logreg_cv.best_params_)
        # print("accuracy :", i, j, ':', logreg_cv.best_score_)

        # SVC Grid search
        SVC_params = {'C': np.arange(1, 10, 0.1), 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
        SVC_cv = GridSearchCV(USVC, SVC_params, cv=3, verbose=0, return_train_score=True)
        SVC_cv.fit(x_train, y_train)
        # print("tuned hyperparameters :(best parameters) ", SVC_cv.best_params_)
        # print("accuracy :", SVC_cv.best_score_)
        # print("res :",logreg_cv.cv_results_)

        logisticRegr2 = LogisticRegression(C=0.511, class_weight=2.001)
        logisticRegr2.fit(x_train, y_train)

        SVC2 = SVC(C=92.101, kernel='rbf', gamma=36.501, class_weight='balanced')
        SVC2.fit(x_train, y_train)

        XGB = XGBClassifier()
        XGB.fit(x_train, y_train)

        estimator = XGBClassifier(objective='binary:logistic', nthread=4, seed=42)
        XGparameters = {'max_depth': range(2, 10, 1), 'n_estimators': range(1, 220, 40),
                        'learning_rate': [0.1, 0.01, 0.05]}
        GXG = GridSearchCV(estimator=estimator, param_grid=XGparameters, scoring='roc_auc', n_jobs=10, cv=3,
                           verbose=False)
        GXG.fit(x_train, y_train)
        # print(grid_search.best_estimator_)

        untrainedScore = metrics.accuracy_score(y_test, logisticRegr.predict(x_test))
        untrainedSum = untrainedSum + untrainedScore
        gridScore = metrics.accuracy_score(y_test, logreg_cv.predict(x_test))
        gridSum = gridSum + gridScore
        TousiScore = metrics.accuracy_score(y_test, logisticRegr2.predict(x_test))
        TousiSum = TousiSum + TousiScore
        scoreCount = scoreCount + 1
        USVCScore = metrics.accuracy_score(y_test, USVC.predict(x_test))
        USVCSum = USVCSum + USVCScore
        GSVCScore = metrics.accuracy_score(y_test, SVC_cv.predict(x_test))
        GSVCSum = GSVCSum + GSVCScore
        TSVCScore = metrics.accuracy_score(y_test, SVC2.predict(x_test))
        TSVCSum = TSVCSum + TSVCScore
        URCScore = metrics.accuracy_score(y_test, URC.predict(x_test))
        URCSum = URCSum + URCScore
        XGBScore = metrics.accuracy_score(y_test, XGB.predict(x_test))
        XGBSum = XGBSum + XGBScore
        GXGScore = metrics.accuracy_score(y_test, GXG.predict(x_test))
        GXGSum = GXGSum + GXGScore

        print('Raw:UntrainedLR:', ts, ":", rTest, ":", j, ':', untrainedScore)
        print('Raw:GridSearchCVLR:', ts, ":", rTest, ":", j, ':', gridScore)
        print('Raw:TousiLR:', ts, ":", rTest, ":", j, ':', TousiScore)
        print('Raw:UntrainedSVC:', ts, ":", rTest, ":", j, ':', USVCScore)
        print('Raw:GridSVC:', ts, ":", rTest, ":", j, ':', GSVCScore)
        print('Raw:TousiSVC:', ts, ":", rTest, ":", j, ':', TSVCScore)
        print('Raw:URC:', ts, ":", rTest, ":", j, ':', URCScore)
        print('Raw:XGB:', ts, ":", rTest, ":", j, ':', XGBScore)
        print('Raw:GXG:', ts, ":", rTest, ":", j, ":", GXGScore)

    untrainedAverage = untrainedSum / scoreCount
    gridAverage = gridSum / scoreCount
    TousiAverage = TousiSum / scoreCount
    USVCAverage = USVCSum / scoreCount
    GSVCAverage = GSVCSum / scoreCount
    TSVCAverage = TSVCSum / scoreCount
    URCAverage = URCSum / scoreCount
    XGBAverage = XGBSum / scoreCount
    GXGAverage = GXGSum / scoreCount

    print('Average:UntrainedLR:', ts, ":", rTest, ":", j, ':', untrainedAverage)
    print('Average:GridSearchCVLR:', ts, ":", rTest, ":", j, ':', gridAverage)
    print('Average:TousiLR:', ts, ":", rTest, ":", j, ':', TousiAverage)
    print('Average:UntrainedSVC:', ts, ":", rTest, ":", j, ':', USVCAverage)
    print('Average:GridSVC:', ts, ":", rTest, ":", j, ':', GSVCAverage)
    print('Average:TousiSVC:', ts, ":", rTest, ":", j, ':', TSVCAverage)
    print('Average:URC:', ts, ":", rTest, ":", j, ':', URCAverage)
    print('Average:XGB:', ts, ":", rTest, ":", j, ':', XGBAverage)
    print('Average:GXG:', ts, ":", rTest, ":", j, ":", GXGAverage)

    scoreDict = {f'untrained LR Average {ts}': untrainedAverage,
                 f'Grid LR Average {ts}': gridAverage,
                 f"Tousi LR Average {ts}": TousiAverage,
                 f"untrained SVC Average {ts}": USVCAverage,
                 f"Grid SVC Average {ts}": GSVCAverage,
                 f"Tousi SVC Average {ts}": TSVCAverage,
                 f"Untrained RC Average {ts}": URCAverage,
                 f"Untrained XG Average {ts}": XGBAverage,
                 f"Grid XG Average {ts}": GXGAverage
                 }
    #scoreMax = max(scoreDict, key=scoreDict.get)
    #print("Maximum value:", ts, ":", scoreMax, scoreDict[scoreMax])

#finalScore[scoreMax] = scoreDict[scoreMax]
#FinalMax = max(finalScore, key=finalScore.get)
#print("Final max:", FinalMax, finalScore[FinalMax])
