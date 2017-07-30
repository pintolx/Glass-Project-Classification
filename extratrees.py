# Feature importance With Extra Trees CLassifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

# Load the data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

# Separate inputs and outputs
X = array[:,0:8]
Y = array[:,8]

# Feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
