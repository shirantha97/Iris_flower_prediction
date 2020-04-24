import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('IRIS.csv')
target = data['species']
species = set()
for values in target:
    species.add(values)
s = list(species)
dataset_rows = list(range(100,150))
data = data.drop(data.index[dataset_rows])
# print(data)
X = data['sepal_width']
Y = data['petal_length']
X_species1 = X[:50]
Y_species1 = Y[:50]
X_species2 = X[50:]
Y_species2 = Y[50:]

plt.scatter(X_species1, Y_species1, color='blue')
plt.scatter(X_species2, Y_species2, color='red')
plt.title('Sepal width,Petal length of Iris-setosa and Iris-versicolor')
plt.legend()
plt.savefig('images/data_setosa&versicolor')
plt.show()
