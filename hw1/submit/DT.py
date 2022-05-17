import  matplotlib.pyplot as plt
import jax as jnp
import numpy as np
import pandas as pd
from collections import Counter,defaultdict

class decisionTree():
    def __init__(self,label):
        self.children = defaultdict()
        self.label = label

def same_label(D):
    return D.iloc[:,-1].unique()[0] if len(D.iloc[:,-1].unique()) == 1 else False

def most_common(D):
    dict = Counter(D.iloc[:,-1])
    return 0 if dict[0] > dict[1] else 1


def gini_gain(D,L):

    def G(D):
        gini = 1
        dict = Counter(D.iloc[:,-1])
        total = sum(dict.values())
        for k,p in dict.items():
            p = p/total
            gini -= p * p
        return gini

    def gini_Info(D,l):
        average_gini = 0
        dict = {0:[0,0],1:[0,0],2:[0,0]}
        for row in D.iterrows():
            value = row[1][l]
            result = row[1][-1]
            dict[value][result] += 1

        for k,v in dict.items():
            partial_gini = 1
            partition_total = sum(v)
            if partition_total == 0:
                continue
            else:
                p0 = v[0]/partition_total
                p1 = v[1]/partition_total
                partial_gini -= (p0*p0 + p1*p1)
            average_gini += partial_gini * partition_total/D.shape[0]

        return average_gini

    G = G(D)

    hash = defaultdict()
    for l in L:
        hash[l] = G - gini_Info(D,l)

    highest_gain = -1
    A = ""
    for k,v in hash.items():
        if v > highest_gain:
            highest_gain = v
            A = k
    return A

def information_gain(D,L):

    def I(D):
        entropy = 0
        dict = Counter(D.iloc[:,-1])
        total = sum(dict.values())
        for k,p in dict.items():
            p = p/total
            entropy -= p * np.log2(p)
        return entropy

    def Info(D,l):
        average_entropy = 0
        dict = {0:[0,0],1:[0,0],2:[0,0]}
        for row in D.iterrows():
            value = row[1][l]
            result = row[1][-1]
            dict[value][result] += 1

        for k,v in dict.items():
            partial_entropy = 0
            partition_total = sum(v)
            if partition_total == 0:
                continue
            else:
                p0 = v[0]/partition_total
                p1 = v[1]/partition_total
                if p0 == 0 or p1 == 0:
                    partial_entropy = 0
                else:
                    partial_entropy -= (p0 * np.log2(p0) + p1 * np.log2(p1))
            average_entropy += partial_entropy * partition_total/D.shape[0]

        return average_entropy

    I = I(D)

    hash = defaultdict()
    for l in L:
        hash[l] = I - Info(D,l)

    highest_gain = -1
    A = ""
    for k,v in hash.items():
        if v > highest_gain:
            highest_gain = v
            A = k
    return A

def createTree(D,L):
    if same_label(D) != False:
        return decisionTree(same_label(D))
    if len(L) == 0:
        return decisionTree(most_common(D))
    A = information_gain(D,L)
    # A = gini_gain(D,L)
    N = decisionTree(A)
    new_L = L.copy()
    new_L.remove(A)
    V = [0,1,2]
    for v in V:
        Dv = D.loc[D[A] == v]
        if len(Dv) == 0:
            return decisionTree(most_common(D))
        Tv = createTree(Dv,new_L)
        N.children[v] = Tv
    return N

def instance_predict(N,instance):
    cur = N
    while len(cur.children) != 0:
        label = cur.label
        value = instance[1][label]
        if value == 0:
            cur = cur.children[0]
        elif value == 1:
            cur = cur.children[1]
        else:
            cur = cur.children[2]
    return cur.label


def fit(N,D):
    accuracy = 0
    i = 0
    for row in D.iterrows():
        if instance_predict(N,row) == row[1][-1]:
            accuracy += 1
    return accuracy/D.shape[0]

def histogram():
    columns = ["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-adi","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-corporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]
    dataset = pd.read_csv("/Users/dttai11/cs589/prj1/DT/house_votes_84.csv")
    train = np.arange(0)
    valid = np.arange(0)
    for i in range(3000):
        dataset = dataset.sample(frac=1)
        validation = dataset.iloc[348:]
        training = dataset.iloc[:348:]
        N = createTree(training,columns)
        print(i)
        training_value = fit(N,training)
        valid_value = fit(N,validation)
        train = np.concatenate((train,[training_value]))
        valid = np.concatenate((valid,[valid_value]))

    plt.hist(train, bins = 24)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title("Training Accuracy")
    print("The mean for training is {}".format(np.mean(train)))
    print("The standard deviation for training is {}".format(np.std(train)))
    plt.show()
    print()
    plt.hist(valid,bins = 24)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title("Testing Accuracy")
    print("The mean for testing is {}".format(np.mean(valid)))
    print("The standard deviation for testing is {}".format(np.std(valid)))
    plt.show()

histogram()

