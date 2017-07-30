# Feature Extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

#Separate inputs and outputs
X = array[:,0:8]
Y = array[:,8]

# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 5)
fit = rfe.fit(X, Y)
print(("Num Features: %s" )%fit.n_features_)
print(("Selected Features: %s") %fit.support_)
print(("Feature Ranking: %s") %fit.ranking_)