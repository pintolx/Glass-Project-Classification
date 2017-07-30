# Leaveone out test
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#Loading the dataset
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

#Split the dataset into inputs and outputs
X = array[:,0:8]
Y = array[:,8]

# Loading the model
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)
print(("Accuracy: %.3f%% (%.3f%%)" %(results.mean()*100, results.std()*100)))