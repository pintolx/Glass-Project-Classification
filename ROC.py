# Cross Validation ROC AUC
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Loading the dataset
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

# Separate inputs and outputs
X = array[:,0:8]
Y = array[:,8]

# Create the model
seed = 7
n_splits = 10
scoring = 'roc_auc'
model = LogisticRegression()
kfold = KFold(n_splits=n_splits, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("AUC: %.3f%% (%.3f%%)") %(results.mean()*100, results.std()*100))