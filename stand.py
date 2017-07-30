# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions

#Loading the data 
filename  = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
# Separate inputs and outputs
X = array[:,0:8]
Y =array[:,8]

#Standardize the data
scaler = StandardScaler()
rescaledX = scaler.fit_transform(X)

#Summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])