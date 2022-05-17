import pandas as pd
from matplotlib import pyplot as plt
from data import cross_validation, columns
import numpy as np

def σ(z):
    return 1 / (1 + np.exp(-z))

def σ_derivative(z):
    return z * (1.0 - z)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class Dense():
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(self.layers)
        self.W = np.array([np.random.normal(0, 0.34, size=(nextLayer_dim, layer_dim+1)) for layer_dim, nextLayer_dim in zip(self.layers[:-1], self.layers[1:])])

    def forwardPropagation(self, x):
        a = x
        a = np.append([1.0], a)
        list = [a]
        for k in range(self.num_layers - 2):
            θ_k = self.W[k]
            z = np.matmul(θ_k, list[k].T)
            a = σ(z)
            a = np.append([1.0], a)
            list.append(a)
        θ = self.W[-1]
        z = np.dot(θ, a)
        a = σ(z)
        list.append(a)
        return list

    def loss(self,X_train,Y_train,λ = 0.0, multiclass = False):
        J = 0
        for x,y in zip(X_train,Y_train):
            if not multiclass:
                f = self.forwardPropagation(x)[-1][0]
                J_i = np.multiply(-y,np.log(f)) - np.multiply(1-y,np.log(1-f))
                J += J_i
            else:
                f = self.forwardPropagation(x)[-1][y-1]
                if f == 0: continue
                J -= np.log(f)

        J = J/X_train.shape[0]
        S = 0
        for W in self.W:
            S += np.sum(np.square(W[1:]))
        S = (λ/(2*X_train.shape[0])) * S
        return J + S

    def backPropagation(self, X_train, Y_train, α=0, λ=.25, multiclass = False, classes = None):
        D = np.array([np.zeros(shape = (nextLayer_dim, layer_dim+1)) for layer_dim, nextLayer_dim in zip(self.layers[:-1], self.layers[1:])])
        for x, y in zip(X_train, Y_train):
            list = self.forwardPropagation(x)
            if multiclass == False:
                δ = np.subtract(list[-1], y)
            else:
                list[-1][y-1] -= 1
                δ = list[-1]
            δ_list = [0]*(self.num_layers - 1)
            δ_list[-1] = δ
            for k in range(self.num_layers - 2, 0, -1):
                a = np.matmul(self.W[k].T,δ_list[k].T)
                b = np.array([σ_derivative(list[k])])
                δ_l = np.multiply(a.T,b)
                δ_l = np.delete(δ_l, 0)
                δ_list[k-1] = δ_l
            for k in range(self.num_layers - 2, -1, -1):
                δ = δ_list[k].reshape(1,len(δ_list[k]))
                a = list[k].reshape(1,len(list[k]))
                j = δ.T * a
                D[k] = np.add(D[k],j)
        for k in range(self.num_layers - 2, -1, -1):
            P = λ * self.W[k]
            P[:,0] = 0
            D[k] = (1 / X_train.shape[0]) * (D[k]+P)
        for k in range(self.num_layers - 2, -1, -1):
            self.W[k] = self.W[k] - np.multiply(α,D[k])

    def evaluate(self, X, Y, multiclass):

        if not multiclass:
            matrix = [[0,0],
                      [0,0]]
        else:
            matrix = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

        for x,y in zip(X,Y):
            a = x
            a = np.append([1.0], a)
            list = [a]
            for k in range(self.num_layers - 2):
                θ_k = self.W[k]
                z = np.matmul(θ_k, list[k].T)
                a = σ(z)
                a = np.append([1.0], a)
                list.append(a)
            θ = self.W[-1]
            z = np.dot(θ, a)
            a = σ(z)
            if not multiclass:
                if y == 0:
                    if a[0] < 0.5: matrix[0][0] += 1
                    else: matrix[0][1] += 1
                elif y == 1:
                    if a[0] > 0.5: matrix[1][1] += 1
                    else: matrix[1][0] += 1
            else:
                a = softmax(a)
                if y == 1:
                    if 0 == np.argmax(a): matrix[0][0] += 1
                    elif 1 == np.argmax(a):matrix[0][1] += 1
                    else: matrix[0][2] += 1
                elif y == 2:
                    if 1 == np.argmax(a): matrix[1][1] += 1
                    elif 0 == np.argmax(a): matrix[1][0] += 1
                    else: matrix[1][2] += 1
                elif y == 3:
                    if 2 == np.argmax(a): matrix[2][2] += 1
                    elif 0 == np.argmax(a): matrix[2][0] += 1
                    else: matrix[2][1] += 1

        if not multiclass:
            acc = (matrix[0][0] + matrix[1][1])/X.shape[0]
            pre = matrix[0][0]/(matrix[0][0] + matrix[0][1]) if (matrix[0][0] + matrix[0][1]) != 0 else 0
            rec = matrix[0][0]/(matrix[0][0] + matrix[1][0]) if (matrix[0][0] + matrix[1][0]) != 0 else 0
            F1 = (2*pre*rec)/(pre+rec) if pre != 0 and rec != 0 else 0
            return acc, F1

        else:
            accuracy = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / X.shape[0]
            precision_1 = matrix[0][0] / (matrix[0][0] + matrix[1][0] + matrix[2][0]) if (matrix[0][0] + matrix[1][0] + matrix[2][0]) != 0 else 0
            recall_1 = matrix[0][0] / (matrix[0][0] + matrix[0][1] + matrix[0][2]) if (matrix[0][0] + matrix[0][1] + matrix[0][2]) != 0 else 0
            F1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if precision_1 != 0 and recall_1 != 0 else 0

            precision_2 = matrix[1][1] / (matrix[1][1] + matrix[0][1] + matrix[2][1]) if (matrix[1][1] + matrix[0][1] + matrix[2][1]) != 0 else 0
            recall_2 = matrix[1][1] / (matrix[1][1] + matrix[1][0] + matrix[1][2]) if (matrix[1][1] + matrix[1][0] + matrix[1][2]) != 0 else 0
            F1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2) if precision_2 != 0 and recall_2 != 0 else 0

            precision_3 = matrix[2][2] / (matrix[2][2] + matrix[0][2] + matrix[1][2]) if (matrix[2][2] + matrix[0][2] + matrix[1][2]) != 0 else 0
            recall_3 = matrix[2][2] / (matrix[2][2] + matrix[2][0] + matrix[2][1]) if (matrix[2][2] + matrix[2][0] + matrix[2][1]) != 0 else 0
            F1_3 = (2 * precision_3 * recall_3) / (precision_3 + recall_3) if precision_3 != 0 and recall_3 != 0 else 0

            F1 = np.mean([F1_1, F1_2, F1_3])

            return accuracy, F1

    def fit(self, X_train, Y_train, validation_data, epochs, α, λ = 0.0, multiclass = False, classes = None, F1 = [], acc = [], loss = [], train_F1_list = [], train_acc_list = []):
        test_loss_list = []
        for epoch in range(1,epochs+1):
            if not multiclass:
                self.backPropagation(X_train,Y_train,α,λ)
            else:
                self.backPropagation(X_train,Y_train,α,λ,True, classes)
            train_acc, train_F1 = self.evaluate(X_train, Y_train, multiclass)
            valid_acc, valid_F1 = self.evaluate(validation_data[0],validation_data[1], multiclass)
            train_loss = self.loss(X_train,Y_train,λ,multiclass)
            valid_loss = self.loss(validation_data[0],validation_data[1],λ,multiclass)
            print("Epoch: {}/{} [======>]".format(epoch,epochs), end = " ")
            print("train_acc {:.6f}".format(train_acc), end = " - ")
            print("train_loss {:.6f}".format(train_loss), end= " - ")
            print("train_F1 {:.6f}".format(train_F1), end = " - ")
            print("val_acc : {:.6f}".format(valid_acc), end = " - ")
            print("val_loss {:.6f}".format(valid_loss), end = " - ")
            print("val_F1 {:.6f}".format(valid_F1))
            test_loss_list.append(valid_loss)
            if epoch == epochs:
                F1.append(valid_F1)
                train_F1_list.append(train_F1)
                acc.append(valid_acc)
                train_acc_list.append(train_acc)
        loss.append(test_loss_list)


def normalize(df):
    result = df.copy()

    def min_max(column):
        return (column - column.min()) / (column.max() - column.min())

    for i  in range(len(df.columns)-1):
        df[df.columns[i]] = min_max(df[df.columns[i]])

    return result

def train(data):
    acc_total = []
    F1_total = []
    loss_list = []
    train_acc_total = []
    train_F1_total = []
    if data == "hw3_house_votes_84.csv":
        df = pd.read_csv("/Users/dttai11/cs589/hw4/datasets/hw3_house_votes_84.csv")
        epochs = 400
    elif data == "hw3_cancer.csv":
        df = pd.read_csv("/Users/dttai11/cs589/hw4/datasets/hw3_cancer.csv", sep="\t")
        epochs = 900
    elif data == 'hw3_wine.csv':
        df = pd.read_csv('/Users/dttai11/cs589/hw4/datasets/hw3_wine.csv', sep='\t')
        epochs = 4000
    else:
        df = pd.read_csv("/Users/dttai11/cs589/hw3/datasets/cmc.data", sep=",", names=columns)
        epochs = 400

    df = df.sample(frac=1)
    df = normalize(df)
    for i in range(8):
        if data == "hw3_house_votes_84.csv":
            X_train, X_test, y_train, y_test = cross_validation(df, i, "class")
            model = Dense(np.array([16, 64, 64, 32, 1]))
            model.fit(X_train, y_train, (X_test, y_test), epochs, 0.4, 0,False,None, F1_total,acc_total, loss_list, train_F1_total, train_acc_total)

        elif data == "hw3_cancer.csv":
            X_train, X_test, y_train, y_test = cross_validation(df, i, "Class")
            model = Dense(np.array([9, 128, 32, 8, 1]))
            model.fit(X_train, y_train, (X_test, y_test), epochs, 0.06, 0.1, False, None, F1_total,acc_total,loss_list, train_F1_total, train_acc_total)

        elif data == 'hw3_wine.csv':
            X_train, X_test, y_train, y_test = cross_validation(df, i, "class")
            y = np.sort(df['class'].unique())
            model = Dense(np.array([13, 128, 3]))
            model.fit(X_train, y_train, (X_test, y_test, y), epochs, 0.003, 0.5, multiclass=True, classes=y,F1 = F1_total, acc = acc_total,loss = loss_list,train_F1_list = train_F1_total, train_acc_list = train_acc_total)

        elif data == "cmc.data":
            X_train, X_test, y_train, y_test = cross_validation(df, i, "method")
            y = np.sort(df['method'].unique())
            model = Dense(np.array([9,256,64,3]))
            model.fit(X_train, y_train, (X_test, y_test, y), epochs, 0.01, 0, multiclass=True, classes=y, F1 = F1_total, acc = acc_total, loss = loss_list, train_F1_list = train_F1_total, train_acc_list = train_acc_total)

        print("End cross_validation {}".format(i+1))
        print("________________________________________________________________________________________________________________________________")
        print()

    F1 = np.mean(F1_total)
    acc = np.mean(acc_total)
    train_acc = np.mean(train_acc_total)
    train_F1 = np.mean(train_F1_total)
    print("Final train_F1 {:.6f}".format(train_F1))
    print("Final val_F1 {:.6f}".format(F1))
    print("Final train_accuracy {:.6f}".format(train_acc))
    print("Final val_accuracy {:.6f}".format(acc))
    loss_list = np.mean(loss_list,axis=0)
    plt.plot([i for i in range(epochs)],loss_list)
    plt.title("valid loss graph")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.show()

data = "hw3_house_votes_84.csv"
train(data)
