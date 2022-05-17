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
#
def most_common(D):
    dict = Counter(D.iloc[:,0])
    return dict.most_common(1)[0][0]

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
        v_partitions = df.iloc[:,A].unique()
        v_partitions.sort()
        hash = {}
        for v in v_partitions:
            hash[v] = [0,0,0]
        for row in df.iterrows():
            answer = row[1][0] - 1
            v = row[1][A]
            hash[v][int(answer)] += 1

        # print(hash)
        average_entropy = 0
        for k,v in hash.items():
            partial_total = sum(v)
            p1,p2,p3 = v[0]/partial_total, v[1]/partial_total, v[2]/partial_total
            # if (p1 == 0 and p2 == 0) or (p1 ==0 and p3 ==0) or (p2 ==0 and p3 == 0):
            #     partial_entropy = 0
            #
            # # if p1 == 0 or p2 == 0 or p3 == 0:
            # #     partial_entropy = 0
            # else:
            I1 = p1 * np.log2(p1) if p1 != 0 else 0
            I2 = p2 * np.log2(p2) if p2 != 0 else 0
            I3 = p3 * np.log2(p3) if p3 != 0 else 0
            partial_entropy = -(I1 + I2 + I3)
            average_entropy += partial_entropy * partial_total/ df.shape[0]

        return average_entropy

    hash = defaultdict(int)
    I = I(df)
    for l in columns:
        if l == 'class':
            continue
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

def createTree(D,L,oL):
    if same_label(D) != False:
        return decisionTree(same_label(D))
    if len(L) == 0:
        return decisionTree(most_common(D))

    A = infoGain(D,L)
    # A = gini_gain(D,L)
    N = decisionTree(A)
    new_L = L.copy()
    new_L.remove(A)
    # V = oL.iloc[:,A].unique()
    V = D.iloc[:,A].unique()

    for v in V:
        Dv = D.loc[D.iloc[:,A] == v]
        if len(Dv) == 0:
            return decisionTree(most_common(D))
        Tv = createTree(Dv,new_L,oL)
        N.children[v] = Tv
    return N

def instance_predict(N,instance,train):
    cur = N
    while len(cur.children) != 0:
        label = cur.label
        value = instance[1][label]
        if value not in cur.children:
            return instance[1]['class']
        cur = cur.children[value]
    return cur.label

def children_most_common(N):
    arr = [v.label for k, v in N.items()]
    dict = Counter(arr)
    return dict.most_common(1)[0][0]

def fit(N,D,train):
    accuracy = 0
    for row in D.iterrows():
        if instance_predict(N,row,train) == row[1]['class']:
            accuracy += 1
    return accuracy/D.shape[0]

def unique_column_value(df):
    arr = []
    for i in df:
        arr.append(np.array(df[i].unique()))
    return arr

class randomForest():
    def __init__(self,df,ntree):
        self.df = df
        self.ntree = ntree
        self.forest = []


    def train(self,columns,k):
        # dataset = self.df.sample(frac=1)
        training, validation = cross_validation(self.df, k)
        for _ in range(self.ntree):
            bootstrapping_training = bootstrapping(training)
            d_tree = createTree(bootstrapping_training, columns, bootstrapping_training.copy())
            self.forest.append(d_tree)
        return self.fit(validation)

    def fit(self,test_set):

        def instance_predict(N, instance):
            cur = N
            while len(cur.children) != 0:
                label = cur.label
                value = instance[1][label]
                if value not in cur.children:
                    return instance[1]['class']
                cur = cur.children[value]
            return cur.label

        pos_count, neg_count = 0, 0
        acc = 0
        for row in test_set.iterrows():
            arr = []
            for tree in self.forest:
                predict_value = instance_predict(tree,row)
                arr.append(int(predict_value))
            arr = np.array(arr)
            most_frequent_value = np.bincount(arr).argmax()
            if most_frequent_value == row[1]['class']:
                # if most_frequent_value == 1:
                #     pos_count += 1
                # elif most_frequent_value == 2:
                #     neg_count += 1
                acc += 1

        # dict = Counter(test_set.iloc[:, -1])
        # neg_test = dict[0]
        # pos_test = dict[1]
        # matrix = {
        #     'True_pos': pos_count,
        #     'True_neg': neg_count,
        #     'False_pos': neg_test - neg_count,
        #     'False_neg': pos_test - pos_count
        # }
        #
        # accuracy = (matrix['True_pos'] + matrix['True_neg']) / (pos_test + neg_test)
        # precision1 = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_pos'])
        # recall1 = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_neg'])
        # F1_1 = (2*precision1*recall1)/(precision1+recall1)
        #
        # precision2 = matrix['True_neg'] / (matrix['True_neg'] + matrix['False_neg'])
        # recall2 = matrix['True_neg'] / (matrix['True_neg'] + matrix['False_pos'])
        # F1_2 = (2*precision2*recall2)/(precision2+recall2)
        #
        # precision = np.mean([precision2,precision1])
        # recall = np.mean([recall1,recall2])
        # F1 = np.mean([F1_1,F1_2])
        # return accuracy, precision, recall, F1
        return acc/test_set.shape[0],0,0,0


def Q2():
    dataset = pd.read_csv('/Users/dttai11/cs589/hw3/datasets/hw3_wine.csv', sep='\t')
    columns = ['class', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    ntrees = [1, 5, 10, 20, 30, 40, 50]
    # ntrees = [1, 5]
    final_acc, final_pre,final_rec,final_f1 = [],[],[],[]
    for ntree in ntrees:
        dataset = dataset.sample(frac=1)
        acc, pre, rec, f1 = [], [], [], []
        for k in range(10):
            forest = randomForest(dataset, ntree)
            accuracy, precision, recall, F1 = forest.train(columns,k)
            acc.append(accuracy)
            # pre.append(precision)
            # rec.append(recall)
            # f1.append(F1)

        final_acc.append(np.mean(acc))
        # final_pre.append(np.mean(pre))
        # final_f1.append(np.mean(f1))
        # final_rec.append(np.mean(rec))
        print(ntree)

    plt.plot(ntrees,final_acc)
    plt.show()
    #
    # plt.plot(ntrees,final_rec)
    # plt.show()
    #
    # plt.plot(ntrees,final_pre)
    # plt.show()
    #
    # plt.plot(ntrees,final_f1)
    # plt.show()

# Q2()