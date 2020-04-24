import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from Model_comparison import comparisons


dataset = pd.read_csv('IRIS.csv')
# shape
print(dataset.shape)
print(dataset.head(10))
# summary
print(dataset.describe())
# class distribution
print(dataset.groupby('species').size())

dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
pyplot.savefig('images/data.value-distribution.png')
pyplot.show()
dataset.hist()
pyplot.savefig('images/data-sistribution.png')
pyplot.show()
scatter_matrix(dataset)
pyplot.savefig('images/matrix.png')
pyplot.show()

arr = dataset.values
X_ratio = 4
Y_ratio = 4
X = arr[:, 0:X_ratio]
Y = arr[:, Y_ratio]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

comparisons(X_train,Y_train)
algorithm = SVC(gamma='auto')
algorithm.fit(X_train, Y_train)
predicted_values = algorithm.predict(X_test)
#generate the accuracy
accuracy = accuracy_score(Y_test,predicted_values)
print(accuracy)

classification = classification_report(Y_test,predicted_values)
print(accuracy)
print(classification)

print('the predict size', len(predicted_values))
print('the y size', len(Y_test))
data_sample = {
    'Y_real' : Y_test,
    'Y_predict' : predicted_values
}
dataFrame = pd.DataFrame(data_sample)
confusionMatrix = pd.crosstab(dataFrame['Y_real'],data_sample['Y_predict'])
print(confusionMatrix)



