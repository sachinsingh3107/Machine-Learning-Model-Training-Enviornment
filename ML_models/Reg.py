import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
# from django.core.files.storage import FileSystemStorage
from sklearn.externals import joblib
from MoTE import settings
from django.core.files import File
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def reg(data, mod, scal):
    print("Reached reg function..")
    path = os.path.join(settings.MEDIA_ROOT, 'model.sav')

    dataset = pd.read_csv(data)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values

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

    if mod == "Linear":
        regressor = lin(xtrain, ytrain)
    elif mod == "DecisionTree":
        regressor = decTree(xtrain, ytrain)
    elif mod == "RandomForest":
        regressor = randFor(xtrain, ytrain)
    elif mod == "Logistic":
        regressor = logReg(xtrain, ytrain)

    if mod == "Polynomial":
        score, regressor = poly(x,y)
    else:
        score = regressor.score(xtest,ytest)
    joblib.dump(regressor, path)
    some_file = open(path, "r")
    dj_file = File(some_file)
    return {'score': score, 'file': dj_file}


def lin(xtrain, ytrain):
    regressor = LinearRegression()
    regressor.fit(xtrain, ytrain)
    return regressor


def poly(x, y, deg=3):
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x_poly, y,test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(xtrain, ytrain)
    return regressor.score(xtest,ytest), regressor


def decTree(xtrain, ytrain):
    regressor = DecisionTreeRegressor()
    regressor.fit(xtrain, ytrain)
    return regressor

def randFor(xtrain, ytrain,n=5):
    regressor = RandomForestRegressor(n_estimators=n)
    regressor.fit(xtrain, ytrain)
    return regressor

def logReg(xtrain, ytrain):
    regressor = LogisticRegression()
    regressor.fit(xtrain, ytrain)
    return regressor

