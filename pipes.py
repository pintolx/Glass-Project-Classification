# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the dataset 
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

# Split the dataset
X = array[:,0:8]
Y = array[:,8]

# Create the pipeline
estimators = []
estimators.append(('Standardize', StandardScaler()))
estimators.append(('LDA', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

# Evaluate the pipeline 
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())