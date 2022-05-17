import pandas as pd
import numpy as np

def cross_validation(df,index,label):
    df = np.array_split(df, 8)
    valid_set = df.pop(index)
    train_set = pd.concat(df)
    X_train, y_train = train_set.drop([label],axis=1).values, train_set[label].values
    X_test, y_test = valid_set.drop([label],axis=1).values, valid_set[label].values
    return X_train, X_test, y_train, y_test

columns = ["Wife's_age", "Wife's_education", "Husband's_education", "children",
                       "Wife's_religion", "Wife's_now_working", "Husband's_occupation", "index",
                       "Media_exposure", "method"]

