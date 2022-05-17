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
        arr.append(df[i].unique())
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
        mean = np.mean(D[l])
        hash = {"low":[0,0],"high":[0,0]}
        for row in D.iterrows():
            answer = row[1][-1]
            v = row[1][l]
            if v >= mean:
                hash["high"][int(answer)] += 1
            elif v < mean:
                hash["low"][int(answer)] += 1

        average_entropy = 0
        for k,v in hash.items():
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

    m = int(np.sqrt(len(L)))
    M = np.random.choice(L, m, replace=False)
    A = information_gain(D, M)
    N = decisionTree(A)
    new_L = list(M).copy()
    new_L.remove(A)

    mean = np.mean(D[A])
    V = [max(D[A]),min(D[A])]
    for v in V:
        if v >= mean:
            Dv = D.loc[D[A] >= mean]
        else:
            Dv = D.loc[D[A] < mean]
        if len(Dv) == 0:
            return decisionTree(most_common(D))
        Tv = createTree(Dv,new_L)
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
                mean = np.mean(np.mean(train[label]))
                if value >= mean:
                    train = train.loc[train[label] >= mean]
                    cur = cur.children["H"]

                else:
                    train = train.loc[train[label] < mean]
                    cur = cur.children["L"]

            return cur.label

        pos_count, neg_count = 0, 0

        for row in test_set.iterrows():
            arr = []
            for tree in self.forest:
                predict_value = instance_predict(tree,row,train)
                arr.append(predict_value)
            arr = np.array(arr)
            most_frequent_value = np.bincount(arr).argmax()
            if most_frequent_value == row[1][-1]:
                if most_frequent_value == 1:
                    pos_count += 1
                elif most_frequent_value == 0:
                    neg_count += 1

        dict = Counter(test_set.iloc[:, -1])
        neg_test = dict[0]
        pos_test = dict[1]
        matrix = {
            'True_pos': pos_count,
            'True_neg': neg_count,
            'False_pos': neg_test - neg_count,
            'False_neg': pos_test - pos_count
        }

        accuracy = (matrix['True_pos'] + matrix['True_neg']) / (pos_test + neg_test)
        precision1 = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_pos'])
        recall1 = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_neg'])
        F1_1 = (2*precision1*recall1)/(precision1+recall1)

        precision2 = matrix['True_neg'] / (matrix['True_neg'] + matrix['False_neg'])
        recall2 = matrix['True_neg'] / (matrix['True_neg'] + matrix['False_pos'])
        F1_2 = (2*precision2*recall2)/(precision2+recall2)

        precision = np.mean([precision2,precision1])
        recall = np.mean([recall1,recall2])
        F1 = np.mean([F1_1,F1_2])
        return accuracy, precision, recall, F1


def Q3():
    dataset = pd.read_csv("/Users/dttai11/cs589/hw3/datasets/hw3_cancer.csv",sep="\t")
    columns = ["Clump_Thickness","Cell_Size_Uniformity","Cell_Shape_Uniformity","Marginal_Adhesion",
               "Single_Epi_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses"]

    ntrees = [1, 5, 10, 20, 30, 40, 50]

    final_acc, final_pre,final_rec,final_f1 = [],[],[],[]
    for ntree in ntrees:
        dataset = dataset.sample(frac=1)
        acc, pre, rec, f1 = [], [], [], []
        for k in range(10):
            forest = randomForest(dataset, ntree)
            accuracy, precision, recall, F1 = forest.train(columns,k)
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

Q3()

