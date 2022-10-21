from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Curently using formatted data from Palacios - Tousi model
x_train, x_test, y_train, y_test = train_test_split(
  ec_x_scl,ec_235, test_size=0.2, random_state=1)
  
len(x_train)
len(x_test)
len(y_train)
len(y_test)

# Recombine because GridSearch performs its own randomized cross validation.
x_all = np.concatenate((x_train,x_test))
len(x_all)
y_all = np.concatenate((y_train,y_test))


# Create untuned models. This serves as the baseline
# Current output:
# LR: 853
# RC: 83.7
# SVM: 88.5
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('Untrained LR SCORE: ',score)
    
RidgeClass = RidgeClassifier()
RidgeClass.fit(x_train, y_train)
predictions = RidgeClass.predict(x_test)
score = RidgeClass.score(x_test, y_test)
print('Untrained RC SCORE: ',score)

SVC = SVC()
SVC.fit(x_train, y_train)
predictions = SVC.predict(x_test)
score = SVC.score(x_test, y_test)
print('Untrained SVC SCORE: ',score)
######################################################
# Tuned LR gets a score of 83.6 (previous 82.7)
# {'C': 9.112475289387755, 'penalty': 'l2'}
parameters_to_tune={"C":np.logspace(0.001,1,100), 
"penalty":["l1","l2",],
"solvers" = ['newton-cg', 'lbfgs', 'liblinear']}
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,parameters_to_tune,cv=5)
logreg_cv.fit(x_all,y_all)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

##################################################################
# Ridge error is 84.0 (previously 83.8)
# Config: {'alpha': 5.26}
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
# Trying to tune ridges
model = RidgeClassifier()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 100, 0.01)
# define search
rc_cv = GridSearchCV(model, grid, cv=5)
# perform the search
results = rc_cv.fit(x_all, y_all)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
