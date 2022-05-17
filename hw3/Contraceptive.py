import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter,defaultdict
from data import cross_validation, bootstrapping


class decisionTree():
    def __init__(self,label):
        self.children = defaultdict()
        self.label = label

def same_label(D):
    return D.iloc[:,-1].unique()[0] if len(D.iloc[:,-1].unique()) == 1 else False

def most_common(D):
    dict = Counter(D.iloc[:,-1])
    return 0 if dict[0] > dict[1] else 1

def unique_column_value(df):
    arr = []
    for i in df:
        arr.append((df[i].unique(),i))
    return arr

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
        if l != "Wife's_age" and l != "children":
            v_partitions = D[l].unique()
            dict = {}
            for v in v_partitions:
                dict[v] = [0,0,0]

            for row in D.iterrows():
                value = row[1][l]
                result = row[1][-1] - 1
                dict[int(value)][int(result)] += 1
        else:
            mean = np.mean(D[l])
            dict = {"low": [0,0,0], "high": [0,0,0]}

            for row in D.iterrows():
                answer = row[1][-1] - 1
                v = row[1][l]
                if v >= mean:
                    dict["high"][int(answer)] += 1
                elif v < mean:
                    dict["low"][int(answer)] += 1

        average_entropy = 0
        for k,v in dict.items():
            partial_total = sum(v)
            if partial_total == 0:
                continue
            p1,p2,p3 = v[0]/partial_total, v[1]/partial_total, v[2]/partial_total
            I1 = p1 * np.log2(p1) if p1 != 0 else 0
            I2 = p2 * np.log2(p2) if p2 != 0 else 0
            I3 = p3 * np.log2(p3) if p3 != 0 else 0
            partial_entropy = -(I1 + I2 + I3)
            average_entropy += partial_entropy * partial_total/ D.shape[0]

        return average_entropy

    I = I(D)

    hash = defaultdict()
    m = int(np.sqrt(len(L)))
    M = list(np.random.choice(L, m, replace=False))
    for l in M:
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

    A = information_gain(D, L)
    N = decisionTree(A)
    new_L = L.copy()
    new_L.remove(A)

    if A != "Wife's_age" and A != "children":
        V = D[A].unique()
        for v in V:
            Dv = D.loc[D[A] == v]
            if len(Dv) == 0:
                return decisionTree(most_common(D))
            Tv = createTree(Dv, new_L)
            N.children[v] = Tv
    else:
        mean = np.mean(D[A])
        V = [min(D[A]), max(D[A])]
        for v in V:
            if v >= mean:
                Dv = D.loc[D[A] >= mean]
            else:
                Dv = D.loc[D[A] < mean]
            if len(Dv) == 0:
                return decisionTree(most_common(D))
            Tv = createTree(Dv, new_L)
            k = "H" if v >= mean else "L"
            N.children[k] = Tv

    return N

def most_common_column(D,l):
    dict = Counter(D[l])
    return dict.most_common(1)[0][0]

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
                if label != "Wife's_age" and label != "children":
                    if value not in cur.children:
                        return most_common_column(self.df,label)
                    cur = cur.children[value]
                else:
                    mean = np.mean(np.mean(train[label]))
                    if value >= mean:
                        train = train.loc[train[label] >= mean]
                        cur = cur.children["H"]
                    else:
                        train = train.loc[train[label] < mean]
                        if "L" not in cur.children:
                            return most_common_column(train, label)
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
            arr = np.array(arr)
            most_frequent_value = np.bincount(arr).argmax()

            if most_frequent_value == row[1][-1]:
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
        F1_1 = (2*precision_1*recall_1)/(precision_1+recall_1) if precision_1 != 0 and recall_1 != 0 else 0

        precision_2 = matrix[1][1] / (matrix[1][1] + matrix[0][1] + matrix[2][1])
        recall_2 = matrix[1][1] / (matrix[1][1] + matrix[1][0] + matrix[1][2])
        F1_2 = (2*precision_2*recall_2)/(precision_2+recall_2) if precision_2 != 0 and recall_2 != 0 else 0

        precision_3 = matrix[2][2] / (matrix[2][2] + matrix[0][2] + matrix[1][2])
        recall_3 = matrix[2][2] / (matrix[2][2] + matrix[2][0] + matrix[2][1])
        F1_3 = (2*precision_3*recall_3)/(precision_3+recall_3) if precision_3 != 0 and recall_3 != 0 else 0

        precision = np.mean([precision_2,precision_1,precision_3])
        recall = np.mean([recall_1,recall_2,recall_3])
        F1 = np.mean([F1_1,F1_2,F1_3])
        return accuracy, precision, recall, F1


def Q4():

    columns = ["Wife's_age","Wife's_education","Husband's_education","children",
               "Wife's_religion","Wife's_now_working","Husband's_occupation","index",
               "Media_exposure","method"]

    dataset = pd.read_csv("/Users/dttai11/cs589/hw3/datasets/cmc.data", sep=",",names = columns)
    ntrees = [1, 5, 10, 20, 30, 40, 50]
    final_acc, final_pre,final_rec,final_f1 = [],[],[],[]
    for ntree in ntrees:
        dataset = dataset.sample(frac=1)
        acc, pre, rec, f1 = [], [], [], []
        for k in range(10):
            forest = randomForest(dataset, ntree)
            accuracy, precision, recall, F1 = forest.train(columns[:-1],k)
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

Q4()

