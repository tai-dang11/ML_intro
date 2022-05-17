
import numpy as np;
import matplotlib.pyplot as plt;
import csv;
import math;

class Data:
    def __init__(self, raw):
        self.result = int(raw[len(raw)-1]);
        self.data = [];
        for i in range(len(raw)-1):
            self.data.append(int(raw[i]));
    def printValue(self):
        for i in self.data:
            print(i);

class DataSet:
    def __init__(self,data , attributes):
        self.data = data;
        self.attributes = attributes;
        #save attribute with index!
    def isSameClass(self):
        cur = self.data[0].result;
        for instance in self.data:
            if instance.result != cur:
                return False;
        return cur;
    def getMajority(self):
        zero = 0; one = 0;
        for instance in self.data:
            if instance.result == 0: zero+=1;
            else: one += 1;
        return 1 if one>zero else 0;
    def getValuesOfAttribute(self, attr):
        [zero, one, two] = [0,1,2];
        return [zero, one, two];
    def shuffle(self):
        np.random.shuffle(self.data);
    def getTrainAndTest(self):
        # self.shuffle();
        train = self.data[:348];
        test = self.data[348:];
        return {
            'train': DataSet(train, self.attributes),
            'test': DataSet(test, self.attributes),
        }

class Node:
    #if already a leaf node
    def __init__(self, attribute=None, label = None):
        self.label = label;
        self.attribute = attribute;
        self.neighbor = [0,0,0];
    def addEdge(self, val, node):
        self.neighbor[val] = node;
    def test(self, data):
        if self.label is not None: return self.label; #leaf node
        valAfterTest = data.data[self.attribute];
        #next node
        return self.neighbor[valAfterTest].test(data);


def decisionTree(D: DataSet, L: []) -> Node:
    isSameClass = D.isSameClass();
    if isSameClass: return Node(label=isSameClass);
    if len(L)==0: return Node(label=D.getMajority());
    #choose best A
    # def giniIndex(l):
    #     def gini(arr):
    #         ps=[];
    #         sum = 0;
    #         for i in arr: sum+=i;
    #         if sum==0: return 1;
    #         for i in arr:
    #             ps.append(i/sum);
    #         final = 0;
    #         for p in ps:
    #             final += p*p;
    #         return 1-final;
    #
    #     V = [[0, 0], [0, 0], [0, 0]];  # count for each values of v, how many 0 and 1 results
    #     for instance in D.data:
    #         V[instance.data[l]][instance.result] += 1;
    #     info = 0;
    #     for arr in V:
    #         info += ((arr[0] + arr[1]) / len(D.data)) * gini(arr);
    #     return info;

    def informationGain(l):
        #return the entropy of D if choose l
        def entropy(arr):
            #calculate the probability and entropy
            ps = [];
            sum = 0;
            for i in arr: sum+=i;
            if sum==0: return 1;
            for i in arr:
                ps.append(i/sum);
            final = 0;
            for p in ps:
                if p!=0:
                    final += -p*math.log2(p);
            return final;
        #go in D, find all instances that have attribute l as 0, 1, or 2. calculate entropy
        V = [[0,0],[0,0],[0,0]]; #count for each values of v, how many 0 and 1 results
        for instance in D.data:
            V[instance.data[l]][instance.result]+=1;
        # print(V)
        info = 0;
        for arr in V:
            info += ((arr[0]+arr[1])/len(D.data))*entropy(arr);
        # print(info)
        return info

    A = -1; lowest = 1;
    for l in L:
        point = informationGain(l);
        if point < lowest:
            A = l;
            lowest = point;
    # print(point)
    N = Node(attribute = A);
    L.remove(A);
    print(L)
    V = D.getValuesOfAttribute(A);
    for v in V:
        instances =  [];
        for instance in D.data:
            if instance.data[A] == v:
                instances.append(instance);
        if len(instances) == 0: return Node(label = D.getMajority());
        # print(DataSet(instances,L.copy()).attributes)
        N.addEdge(v, decisionTree(DataSet(instances,L.copy()), L.copy()));
    return N;

reader = csv.reader(open('/Users/dttai11/cs589/prj1/DT/house_votes_84.csv'));
processed= [];
for row in reader:
    processed.append(row);
instances = []
for i in range(1, len(processed)):
    instances.append(Data(processed[i]));

originalData = DataSet(instances, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);

def plotTrainingAccuracy():
    TEST_TIME = 1;
    accuracy = [];
    for time in range(TEST_TIME):
        trainAndTest = originalData.getTrainAndTest();
        train = trainAndTest['train'];
        node = decisionTree(train, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        countTrue = 0;
        for i in range(len(train.data)):
            if node.test(train.data[i]) == train.data[i].result:
                countTrue += 1;
        accuracy.append(100 * countTrue / (len(train.data)));
    # print('Average Accuracy: ', np.average(accuracy));
    # print('Standard Deviation: ', np.std(accuracy));
    # plt.hist(accuracy, bins = 16);
    # plt.ylabel('Frequency');
    # plt.xlabel('Accuracy');
    # plt.show();

# plotTrainingAccuracy();

def plotTestingAccuracy():
    TEST_TIME = 1000;
    accuracy = [];
    for time in range(TEST_TIME):
        trainAndTest = originalData.getTrainAndTest();
        train = trainAndTest['train'];
        test = trainAndTest['test'];
        node = decisionTree(train, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        countTrue = 0;
        for i in range(len(test.data)):
            if node.test(test.data[i]) == test.data[i].result:
                countTrue += 1;
        accuracy.append(100 * countTrue / (len(test.data)));
    print('Average Accuracy: ', np.average(accuracy));
    print('Standard Deviation: ', np.std(accuracy));
    plt.hist(accuracy, bins = 16);
    plt.ylabel('Frequency');
    plt.xlabel('Accuracy');
    plt.show();
#
#
#
#

x = [[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 9, 10, 11, 12, 13, 14, 15],
[7, 9, 10, 11, 12, 13, 14, 15],
[9, 10, 11, 12, 13, 14, 15],
[10, 11, 12, 13, 14, 15],
[11, 12, 13, 14, 15],
[12, 13, 14, 15],
[13, 14, 15],
[14, 15],
[15],
[],
[6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[0, 1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]



y =[
[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 9, 10, 11, 12, 13, 14, 15],
[7, 9, 10, 11, 12, 13, 14, 15],
[9, 10, 11, 12, 13, 14, 15],
[10, 11, 12, 13, 14, 15],
[11, 12, 13, 14, 15],
[12, 13, 14, 15],
[13, 14, 15],
[14, 15],
[15],
[],
[6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[0, 1, 4, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[4, 5, 6, 7, 8, 9, 10, 12, 13, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[7, 8, 9, 10, 11, 12, 13, 14, 15],
[8, 9, 10, 11, 12, 13, 14, 15],
[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15],
[0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 15],
[0, 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14,15],
[1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[0, 1, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
[1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]]

print(x==y)