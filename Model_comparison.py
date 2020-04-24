from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def comparisons(X, Y):
    print("The comparison between models using the training dataset")
    algorithms = [('logistic regression', LogisticRegression(solver='liblinear', multi_class='ovr')),
                  ('linear discrimant analaysis', LinearDiscriminantAnalysis()),
                  ('K Nearest Neighbors', KNeighborsClassifier()),
                  ('Desion tree classsifiert', DecisionTreeClassifier()),
                  ('Support Vector Machine', SVC(gamma='auto'))]
    algorithm_accuracy = []

    for name, algorithm in algorithms:
        stratifiedKFold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        crossval_score = cross_val_score(algorithm, X, Y, cv=stratifiedKFold, scoring='accuracy')
        algorithm_accuracy.append(crossval_score)
        print(name)
        print(crossval_score.mean())
        print(crossval_score.std())

    plt.boxplot(algorithm_accuracy)
    plt.title('Machine learning models')
    plt.savefig('images/algorithms-comparison.png')
    plt.show()
