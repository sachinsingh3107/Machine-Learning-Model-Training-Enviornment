import pandas as pd
from sklearn.preprocessing import Imputer



def getTitles(data):
    df = pd.read_csv(data)
    lis = list(df)
    return lis


def getFea(data,lisf,lisl):
    df = pd.read_csv(data)
    dff = df[[col for col in lisf]]
    dfl = df[[col for col in lisl]]
    df = dff.join(dfl)
    df.to_csv(data,index=False)


def dumm(data, lis):
    df = pd.read_csv(data)
    for col in lis:
        df = pd.get_dummies(df, columns=[col]).iloc[:,:-1]

    d = df.values
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer = imputer.fit(d[:, :])
    d[:, :] = imputer.transform(d[:, :])


    dataframe = pd.DataFrame(d,columns=list(df))
    dataframe.to_csv(data, index=False)

