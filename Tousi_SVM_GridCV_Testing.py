from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# This will split into a test and train set. I don't have a non-randomization
# interval, so I'm using one as mine I guess?
X_train, X_test = train_test_split(
  X_Ec_scl_126_sc_5f, test_size=0.2, random_state=1)
#print(X_train)
# I want to make sure it is 80% train and 20% test
len(X_train)
len(X_test)


# No. Randomization is going to make the X and Y go mucky. 
# Start from the top with XY data?
in_E_np # EC is the last column so the formatting still works.


FES_Ec_126_sc_5f = np.delete(in_E_np, [], axis=1)
FES_Ec_126_sc_5f[:, 13]  = np.where(FES_Ec_126_sc_5f[:, 13] > 126, 1, 0)

X_FES_Ec_126_sc_5f = FES_Ec_126_sc_5f[:,0:13]
Y_FES_Ec_126_sc_5f = FES_Ec_126_sc_5f[:,13]
Y_FES_Ec_126_sc_5f=Y_FES_Ec_126_sc_5f.astype('int')

scalerX_FES_Ec_126_sc_5f = MinMaxScaler()
X_FES_Ec_126_sc_5f_Scaled = scalerX_FES_Ec_126_sc_5f.fit_transform(X_FES_Ec_126_sc_5f)

X_train, X_test, y_train, y_test = train_test_split(
  X_FES_Ec_126_sc_5f_Scaled, Y_FES_Ec_126_sc_5f, test_size=0.2, random_state=2)
# According to the paper, the C value should be between 1 and 41 
# and gammais between 1 and 51
len(X_train)
len(X_test)
len(y_train)
len(y_test)


# Why this no work?
# I use only rbf because it is mentioned in the paper.

# Working on FES: C = 0.3 and gamma = 5.5 apparently. 
parameters = {'C': [0.1,0.2, 0.3, 0.33, 1, 10, 100], 'gamma': [1, 0.1,0.01,0.001, 2, 3, 4, 5, 5.5],'kernel': ['rbf']}
grid_obj = GridSearchCV(SVC(), parameters, refit=True,verbose=2, error_score='raise')
grid_obj.fit(X_train,y_train)

# print best parameter after tuning
print(grid_obj.best_params_)
  
# print how our model looks after hyper-parameter tuning
print(grid_obj.best_estimator_)


