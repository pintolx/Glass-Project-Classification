#Normalize the data
from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions

#Load the dataset
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values

#Separate inputs and outputs
X = array[:,0:8]
Y = array[:,8]

#Normalize the data
scaler = Normalizer()
rescaledX = scaler.fit_transform(X)
#Summarose the transfromed data 
set_printoptions(precision=3)
print(rescaledX[:5,:])