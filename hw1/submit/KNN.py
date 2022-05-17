import sys
import  matplotlib.pyplot as plt
import jax
import numpy as np
import pandas as pd
from collections import Counter,defaultdict


def euclidean_distance(i1,i2,num):
    dist = 0
    for i in range(num):
        dist += (i1[i]-i2[i])**2
    return dist**0.5

def normalize(df):
    result = df.copy()

    def min_max(column):
        return (column - column.min()) / (column.max() - column.min())

    for i  in range(len(df.columns)-1):
        df[df.columns[i]] = min_max(df[df.columns[i]])

    return result

def nearest_neighbors(trainng_data,x,num_atrributes,k):
    near_arr = []
    for iris in trainng_data.iterrows():
        near_arr.append([iris[0],euclidean_distance(iris[1],x,num_atrributes),iris[1][4]])
    near_arr.sort(key = lambda e:e[1])
    k_neighbors = defaultdict(list)
    for i in range(k):
        k_neighbors[near_arr[i][2]].append(near_arr[i][0])
    count_neightbors = Counter(k_neighbors)
    return count_neightbors

def knn(data, x, k):
    neighbors = nearest_neighbors(data,x,4,k)
    majority_label = ""
    majority_num = float("-inf")
    for k,v in neighbors.items():
        if len(v) > majority_num:
            majority_label = k
            majority_num = len(v)
    if x[4] == majority_label:
        return True
    return False


def train(data,k,size,test = False,test_data = None, test_size = 0):
    if size <= k:
        print("data size is smaller than k")
        sys.exit()
    train_correct_predicted = 0
    test_correct_predicted = 0
    for iris in data.iterrows():
        if knn(data,iris[1],k):
            train_correct_predicted += 1
    if test:
        for iris_test in test_data.iterrows():
            if knn(data,iris_test[1],k):
                test_correct_predicted += 1
        return train_correct_predicted/size, test_correct_predicted/test_size
    return train_correct_predicted / size

def graph():
    dataset = pd.read_csv("Iris/iris.csv", header=None,
                          names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Type'])

    dataset = normalize(dataset)
    train_arr = np.arange(0)
    valid_arr = np.arange(0)

    for k in range(1,51,2):

        train_repeat = np.arange(0)
        valid_repeat = np.arange(0)

        for i in range(20):

            dataset = dataset.sample(frac=1)
            validation = dataset.iloc[120:]
            training = dataset.iloc[:120:]
            # train_value = train(training,k,training.shape[0],test=False)
            train_value,valid_value = train(training,k,training.shape[0],test=True,test_data=validation,test_size = validation.shape[0])

            train_repeat = np.append(train_repeat,[train_value])

            valid_repeat = np.append(valid_repeat,[valid_value])
        train_arr = np.concatenate((train_arr,train_repeat))
        valid_arr = np.concatenate((valid_arr,valid_repeat))

    train_arr = np.reshape(train_arr,(25,20))

    valid_arr = np.reshape(valid_arr, (25,20))

    mean_arr,error_arr = np.arange(0),np.arange(0)

    for arr in train_arr:
        mean_arr = np.append(mean_arr,[np.mean(arr)])
        error_arr = np.append(error_arr,[np.std(arr)])

    test_mean_arr, test_error_arr = np.arange(0),np.arange(0)

    for tarr in valid_arr:
        test_mean_arr = np.append(test_mean_arr,[np.mean(tarr)])
        test_error_arr = np.append(test_error_arr,[np.std(tarr)])

    return [mean_arr,error_arr],[test_mean_arr, test_error_arr]

def test_graph():
    train,test = graph()
    k_range = np.arange(1,51,2)
    for i in range(2):
        if i == 0:
            fig, ax = plt.subplots()
            plt.xlabel('K value')
            plt.ylabel('Accuracy')
            plt.title("Testing Accuracy")
            ax.errorbar(k_range, test[0],
                        yerr=test[1],
                        fmt='-o')
        else:
            fig, ax = plt.subplots()
            plt.xlabel('K value')
            plt.ylabel('Accuracy')
            plt.title("Training Accuracy")
            ax.errorbar(k_range, train[0],
                        yerr=train[1],
                        fmt='-o')
            plt.show()

test_graph()