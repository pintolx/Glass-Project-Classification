# Cross Validation for Regression Example
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Loading the dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSAT', 'MDEV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values

# Split the dataset 
X = array[:,0:13]
Y = array[:,13]

# Creating the model 
seed = 7
n_splits = 10
model = LinearRegression()
kfold = KFold(n_splits=n_splits, random_state=7)
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(("MAE: %.3f%% (%.3f%%)") %(results.mean(), results.std()))