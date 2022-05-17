from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from data import cross_validation, bootstrapping

class decisionTree():
    def __init__(self,label):
        self.children = defaultdict()
        self.label = label

def same_label(D):
    return D['class'].unique()[0] if len(D['class'].unique()) == 1 else False

def most_common(D):
    dict = Counter(D.iloc[:,0])
    return dict.most_common(1)[0][0]

def Gini(df,columns):
    def G(df):
        gini = 1
        dict = Counter(df['class'])
        total = sum(dict.values())
        for k, p in dict.items():
            p = p / total
            gini -= p * p
        return gini

    def gini_Info(df, A):
        mean = np.mean(df.iloc[:,A])
        hash = {"low":[0,0,0],"high":[0,0,0]}
        for row in df.iterrows():
            answer = row[1][0] - 1
            v = row[1][A]
            if v >= mean:
                hash["high"][int(answer)] += 1
            elif v < mean:
                hash["low"][int(answer)] += 1

        average_gini = 0
        for k,v in hash.items():
            partial_gini = 1
            partial_total = sum(v)
            if sum(v) == 0:
                continue
            else:
                p1,p2,p3 = v[0]/partial_total, v[1]/partial_total, v[2]/partial_total
                partial_gini -= (p1*p1 + p2*p2 + p3*p3)
            average_gini += partial_gini * partial_total/ df.shape[0]

        return average_gini

    hash = defaultdict(int)
    I = G(df)
    m = int(np.sqrt(len(columns)))
    M = list(np.random.choice(columns, m, replace=False))
    for l in M:
        hash[l] = I - gini_Info(df,l)

    A, highest_gain = -1,-1
    for k,v in hash.items():
        if v > highest_gain:
            highest_gain = v
            A = k
    return A


def infoGain(df,columns):
    def I(df):
        hash = Counter(df['class'])
        I = 0
        total = sum(hash.values())
        for k,v in hash.items():
            v = v/total
            I += v * np.log2(v)
        return -I

    def info_A(df, A):
        mean = np.mean(df.iloc[:,A])
        hash = {"low":[0,0,0],"high":[0,0,0]}
        for row in df.iterrows():
            answer = row[1][0] - 1
            v = row[1][A]
            if v >= mean:
                hash["high"][int(answer)] += 1
            elif v < mean:
                hash["low"][int(answer)] += 1

        average_entropy = 0
        for k,v in hash.items():
            partial_total = sum(v)
            if sum(v) == 0:
                # partial_entropy = 0
                continue
            else:
                p1,p2,p3 = v[0]/partial_total, v[1]/partial_total, v[2]/partial_total
                I1 = p1 * np.log2(p1) if p1 != 0 else 0
                I2 = p2 * np.log2(p2) if p2 != 0 else 0
                I3 = p3 * np.log2(p3) if p3 != 0 else 0
                partial_entropy = -(I1 + I2 + I3)
            average_entropy += partial_entropy * partial_total/ df.shape[0]

        return average_entropy

    hash = defaultdict(int)
    I = I(df)
    m = int(np.sqrt(len(columns)))
    M = list(np.random.choice(columns, m, replace=False))
    for l in M:
        hash[l] = I - info_A(df,l)

    A, highest_gain = -1,-1
    for k,v in hash.items():
        if v > highest_gain:
            highest_gain = v
            A = k
    return A

def most_common_column(D,l):
    dict = Counter(D.iloc[:,l])
    return dict.most_common(1)[0][0]

def createTree(D,L):
    if same_label(D) != False:
        return decisionTree(same_label(D))
    if len(L) == 0:
        return decisionTree(most_common(D))
    # A = infoGain(D,L)
    A = Gini(D,L)
    N = decisionTree(A)
    new_M = L.copy()
    new_M.remove(A)
    mean = np.mean(D.iloc[:,A])
    V = [max(D.iloc[:,A]),min(D.iloc[:,A])]
    for v in V:
        if v >= mean:
            Dv = D.loc[D.iloc[:, A] >= mean]
        else:
            Dv = D.loc[D.iloc[:, A] < mean]
        if len(Dv) == 0:
            return decisionTree(most_common(D))
        Tv = createTree(Dv,new_M)
        k = "H" if v >= mean else "L"
        N.children[k] = Tv
    return N



class randomForest():
    def __init__(self,df,ntree):
        self.df = df
        self.ntree = ntree
        self.forest = []


    def train(self,columns,k):
        training, validation = cross_validation(self.df, k)
        for _ in range(self.ntree):
            bootstrapping_training = bootstrapping(training)
            d_tree = createTree(bootstrapping_training, columns)
            self.forest.append(d_tree)
        return self.fit(validation)

    def fit(self,test_set):

        train = self.df.copy()
        def instance_predict(N, instance,train):
            cur = N
            while len(cur.children) != 0:
                label = cur.label
                value = instance[1][label]
                mean = np.mean(np.mean(train.iloc[:, label]))
                if value >= mean:
                    train = train.loc[train.iloc[:, label] >= mean]
                    cur = cur.children["H"]
                else:
                    train = train.loc[train.iloc[:, label] < mean]
                    cur = cur.children["L"]
            return cur.label

        matrix = [
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
        for row in test_set.iterrows():
            arr = []
            for tree in self.forest:
                predict_value = instance_predict(tree,row,train)
                arr.append(predict_value)
            arr = list(np.array(arr))
            most_frequent_value = np.bincount(arr).argmax()
            if most_frequent_value == row[1][0]:
                if most_frequent_value == 1:
                    matrix[0][0] += 1
                elif most_frequent_value == 2:
                    matrix[1][1] += 1
                else:
                    matrix[2][2] += 1
            else:
                if most_frequent_value == 1:
                    if row[1][-1] == 2:
                        matrix[0][1] += 1
                    else:
                        matrix[0][2] += 1
                elif most_frequent_value == 2:
                    if row[1][-1] == 1:
                        matrix[1][0] += 1
                    else:
                        matrix[1][2] += 1
                else:
                    if row[1][-1] == 1:
                        matrix[2][0] += 1
                    else:
                        matrix[2][1] += 1

        accuracy = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / test_set.shape[0]
        precision_1 = matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0])
        recall_1 = matrix[0][0] / (matrix[0][0] + matrix[0][1] + matrix[0][2])
        F1_1 = (2*precision_1*recall_1)/(precision_1+recall_1)

        precision_2 = matrix[1][1] / (matrix[1][1] + matrix[0][1] + matrix[2][1])
        recall_2 = matrix[1][1] / (matrix[1][1] + matrix[1][0] + matrix[1][2])
        F1_2 = (2*precision_2*recall_2)/(precision_2+recall_2)

        precision_3 = matrix[2][2] / (matrix[2][2] + matrix[0][2] + matrix[1][2])
        recall_3 = matrix[2][2] / (matrix[2][2] + matrix[2][0] + matrix[2][1])
        F1_3 = (2*precision_3*recall_3)/(precision_3+recall_3)

        precision = np.mean([precision_2,precision_1,precision_3])
        recall = np.mean([recall_1,recall_2,recall_3])
        F1 = np.mean([F1_1,F1_2,F1_3])
        return accuracy, precision, recall, F1


def Q2():
    dataset = pd.read_csv('/Users/dttai11/cs589/hw3/datasets/hw3_wine.csv', sep='\t')
    columns = ['class', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    ntrees = [1, 5, 10, 20, 30, 40, 50]
    final_acc, final_pre,final_rec,final_f1 = [],[],[],[]
    for ntree in ntrees:
        dataset = dataset.sample(frac=1)
        acc, pre, rec, f1 = [], [], [], []
        for k in range(10):
            forest = randomForest(dataset, ntree)
            accuracy, precision, recall, F1 = forest.train(columns[1:],k)
            acc.append(accuracy)
            pre.append(precision)
            rec.append(recall)
            f1.append(F1)

        final_acc.append(np.mean(acc))
        final_pre.append(np.mean(pre))
        final_f1.append(np.mean(f1))
        final_rec.append(np.mean(rec))
        print(ntree)

    plt.plot(ntrees,final_acc)
    plt.title("Accuracy graph")
    plt.xlabel("ntree values")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(ntrees,final_rec)
    plt.title("Recall graph")
    plt.xlabel("ntree values")
    plt.ylabel("Recall")
    plt.show()

    plt.plot(ntrees,final_pre)
    plt.title("Precision graph")
    plt.xlabel("ntree values")
    plt.ylabel("Precision")
    plt.show()

    plt.plot(ntrees,final_f1)
    plt.title("F1 graph")
    plt.xlabel("ntree values")
    plt.ylabel("F1")
    plt.show()

Q2()


