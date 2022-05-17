import pandas as pd
import numpy as np

def cross_validation(df,index):
    df = np.array_split(df, 10)
    valid_set = df.pop(index)
    train_set = pd.concat(df)
    return train_set,valid_set

def bootstrapping(df):
    columns = list(df.columns)
    bootstrapped_data = pd.DataFrame(columns = columns)
    for _ in range(df.shape[0]):
        df1 = df.sample()
        bootstrapped_data = pd.concat([bootstrapped_data, df1])
    return bootstrapped_data
