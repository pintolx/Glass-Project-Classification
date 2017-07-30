# View the first 20 rows of the dataset
from pandas import read_csv
from pandas import set_option
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
shape = data.shape
type = data.dtypes
description = data.describe()
class_counts = data.groupby('class').size()
set_option("display.width", 100)
set_option('precision', 3)
correlation = data.corr(method='pearson')
skew = data.skew()
print(shape)
print(peek)
print(type)
print(description)
print(class_counts)
print(correlation)
print(skew)