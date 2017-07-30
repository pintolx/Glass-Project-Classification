# Feature Extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA

# Load the data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values

#Separate the inputs from outputs
X = array[:,0:8]
Y = array[:,8]

#Feature selection
pca = PCA(n_components=3)
fit = pca.fit(X)
rescaledX = fit.transform(X)
#Summarise components
print(("Explained Variance: %s") %fit.explained_variance_ratio_)
print(fit.components_)
print(rescaledX[:5,:])