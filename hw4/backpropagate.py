
import numpy as np

def σ(z):
    return 1 / (1 + np.exp(-z))

def σ_derivative(z):
    return z * (1.0 - z)

def forwardPropagation(x):
    a = x
    a = np.append([1.0], a)
    list = [a]
    for k in range(num_layers - 2):
        θ_k = W[k]
        l = list[k].T
        z = np.matmul(θ_k, l)
        a = σ(z)
        a = np.append([1.0], a)
        list.append(a)
    θ = W[-1]
    z = np.dot(θ, a)
    a = σ(z)
    list.append(a)
    print("forward propagation")
    print(list)
    print()
    return list


def backPropagation(X_train, Y_train, λ = 0.0, multiclass=False, classes=None):
    D = np.array([np.zeros(shape=(nextLayer_dim, layer_dim + 1)) for layer_dim, nextLayer_dim in
                  zip(layers[:-1], layers[1:])])
    for x, y in zip(X_train, Y_train):
        list = forwardPropagation(x)
        if multiclass == False:
            δ = np.subtract(list[-1], y)
        else:
            list[-1][y - 1] -= y
            δ = list[-1]
        δ_list = [0] * (num_layers - 1)
        δ_list[-1] = δ
        for k in range(num_layers - 2, 0, -1):
            a = np.matmul(W[k].T, δ_list[k].T)
            b = np.array([σ_derivative(list[k])])
            δ_l = np.multiply(a.T, b)
            δ_l = np.delete(δ_l, 0)
            δ_list[k - 1] = δ_l
        print("delta lists")
        print(δ_list)
        print()
        for k in range(num_layers - 2, -1, -1):
            δ = δ_list[k].reshape(1, len(δ_list[k]))
            a = list[k].reshape(1, len(list[k]))
            j = δ.T * a
            print("Gradients of Theta")
            print(j)
            print()
            D[k] = np.add(D[k], j)


    for k in range(num_layers - 2, -1, -1):
        P = λ * W[k]
        P[:, 0] = 0
        D[k] = (1 / X_train.shape[0]) * (D[k] + P)

    print()
    print("Final regularized gradients of all thetas")
    for i in range(len(D)):
        print("theta {}".format(i+1))
        print(D[i])
        print()


# example 1
layers = [1,2,1]
num_layers = len(layers)
W = [np.array([[0.4, 0.1], [0.3, 0.2]]), np.array([[0.7, 0.5, 0.6]])]
x = np.array([[0.13],[0.42]])
y = np.array([[0.9],[0.23]])
backPropagation(x,y,0.0)

# example 2
# layers = [2, 4, 3, 2]
# num_layers = len(layers)
#
# x = np.array([[.32,.68],[0.83, 0.02]])
# y = np.array([[0.75,0.98],[0.75,0.28]])
#
# W = [np.array([[0.42, 0.15,.4],
#                [0.72,.1,.54],
#                [0.01,.19,.42],
#                [0.3,.35,.68]]),
#
#         np.array([[0.21,0.67, 0.14,.96,.87],
#                 [0.87,0.42, 0.2,.32,.89],
#                 [0.03,0.56, 0.8,.69,.09]]),
#
#         np.array([[0.04,0.87, 0.42, 0.53],
#                   [0.17,0.1, 0.95, 0.69]])
# ]
#
# backPropagation(x,y,0.25)