# Cross Validating Mean Squared Error
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Loading the dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSAT', 'MDEV']
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values

#Split the inputs and outputs
X = array[:,0:13]
Y = array[:,13]

# Designing the model
seed = 7
n_splits = 10
socring = 'neg_mean_squared_error'
kfold = KFold(n_splits=n_splits, random_state=seed)
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold, scoring=socring)
print(("MSE: %.3f (%.3f)") %(results.mean(), results.std()))
