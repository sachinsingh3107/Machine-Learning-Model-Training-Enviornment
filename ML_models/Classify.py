import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


def classify(data, mod, scal):
    print("class entred")
    dataset = pd.read_csv(data)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    if scal == "MinMax":
        sc = MinMaxScaler()
        x = sc.fit_transform(x)
    elif scal == "Normalizer":
        sc = Normalizer()
        x = sc.fit_transform(x)
    elif scal == "Standard":
        sc = StandardScaler()
        x = sc.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    if mod == "DecisionTree":
        regressor = decTree(xtrain,ytrain)
    elif mod == "RandomForest":
        regressor = randFor(xtrain,ytrain)
    elif mod == "KNN":
        regressor = knClass(xtrain,ytrain)
    elif mod == "SVM":
        regressor = svClass(xtrain,ytrain)
    elif mod == "NaiveBayes":
        regressor = naivBay(xtrain,ytrain)

    # joblib.dump(regressor, "../media/model.sav")
    return {'score': regressor.score(xtest,ytest)}


def decTree(xtrain, ytrain):
    regressor = DecisionTreeClassifier()
    regressor.fit(xtrain, ytrain)
    return regressor


def randFor(xtrain, ytrain, n=5):
    regressor = RandomForestClassifier(n_estimators=n)
    regressor.fit(xtrain, ytrain)
    return regressor


def svClass(xtrain, ytrain):
    regressor = SVC()
    regressor.fit(xtrain, ytrain)
    return regressor


def knClass(xtrain, ytrain):
    regressor = KNeighborsClassifier()
    regressor.fit(xtrain, ytrain)
    return regressor


def naivBay(xtrain, ytrain):
    regressor = GaussianNB()
    regressor.fit(xtrain, ytrain)
    return regressor
