import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('breast-cancer-wisconsin-data.csv')


#Get Target data 
y = data['diagnosis']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['id','diagnosis','Unnamed: 32'], axis = 1)

logModel = LogisticRegression()

param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(logModel, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)

best_clf = clf.fit(X,y)

best_clf.best_estimator_

https://www.kaggle.com/code/funxexcel/p2-logistic-regression-hyperparameter-tuning/notebook
