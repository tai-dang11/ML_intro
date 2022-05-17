from utils import *
from collections import defaultdict
import jax.numpy as jnp
import random
import numpy as np
import matplotlib.pyplot as plt

class naive_bayes():
    def __init__(self, pos_data, neg_data, words, alpha):
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.words = words
        self.pos_word_frequency = defaultdict(int)
        self.neg_word_frequency = defaultdict(int)
        self.alpha = alpha
        self.count_prob()

    def total(self):
        return len(self.pos_data) + len(self.neg_data)

    def neg_instance(self):
        return len(self.neg_data) / self.total()

    def pos_instance(self):
        return len(self.pos_data) / self.total()

    def count_prob(self):
        for doc in self.pos_data:
            for word in doc:
                self.pos_word_frequency[word] += 1

        for doc in self.neg_data:
            for word in doc:
                self.neg_word_frequency[word] += 1

        total_pos = sum(self.pos_word_frequency.values()) + self.alpha * len(self.words)
        total_neg = sum(self.neg_word_frequency.values()) + self.alpha * len(self.words)

        for word in self.words:
            self.pos_word_frequency[word] = (self.pos_word_frequency[word]  + self.alpha) / total_pos
            self.neg_word_frequency[word] = (self.neg_word_frequency[word] + self.alpha) / total_neg

    def train_fit(self, test_data):
        pos_probability = self.pos_instance()
        neg_probability = self.neg_instance()

        for word in test_data:
            pos_probability *= self.pos_word_frequency[word]
            neg_probability *= self.neg_word_frequency[word]

        if pos_probability > neg_probability:
            return 1
        elif neg_probability > pos_probability:
            return -1
        else:
            return random.randrange(-1, 2, 2)

    def train_log_fit(self, test_data):

        pos_probability = np.log(self.pos_instance())
        neg_probability = np.log(self.neg_instance())

        for word in test_data:
            pos_word_prob = self.pos_word_frequency[word] if self.pos_word_frequency[word] != 0 else np.finfo(float).eps/sum(self.pos_word_frequency.values())
            neg_word_prob = self.neg_word_frequency[word] if self.neg_word_frequency[word] != 0 else np.finfo(float).eps/sum(self.neg_word_frequency.values())
            pos_probability += np.log(pos_word_prob)
            neg_probability += np.log(neg_word_prob)

        if pos_probability > neg_probability:
            return 1
        elif neg_probability > pos_probability:
            return -1
        else:
            return random.randrange(-1, 2, 2)

    def evaluate(self, pos_test, neg_test, log=False):
        pos_count, neg_count = 0, 0
        if log == False:
            for pos_ins in pos_test:
                if self.train_fit(pos_ins) == 1:
                    pos_count += 1
            for neg_ins in neg_test:
                if self.train_fit(neg_ins) == -1:
                    neg_count += 1
        else:
            for pos_ins in pos_test:
                if self.train_log_fit(pos_ins) == 1:
                    pos_count += 1
            for neg_ins in neg_test:
                if self.train_log_fit(neg_ins) == -1:
                    neg_count += 1

        matrix = {
            'True_pos': pos_count,
            'True_neg': neg_count,
            'False_pos': len(neg_test) - neg_count,
            'False_neg': len(pos_test) - pos_count
        }

        accuracy = (matrix['True_pos'] + matrix['True_neg']) / (len(pos_test) + len(neg_test))
        precision = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_pos'])
        recall = matrix['True_pos'] / (matrix['True_pos'] + matrix['False_neg'])
        print('The confusion matrix is', matrix)
        print('The accuracy is ', accuracy)
        print('The precision is', precision)
        print('The recall is ', recall)
        print()
        return accuracy

def load_data(percentage_positive_instances_train,percentage_negative_instances_train,percentage_positive_instances_test,percentage_negative_instances_test):

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)
    # print("Number of positive training instances:", len(pos_train))
    # print("Number of negative training instances:", len(neg_train))
    # print("Number of positive test instances:", len(pos_test))
    # print("Number of negative test instances:", len(neg_test))
    return (pos_train, neg_train, vocab),(pos_test, neg_test)


def naive_bayes1():
    (pos_train, neg_train, vocab), (pos_test, neg_test) = load_data(0.2,0.2,0.2,0.2)
    model1 = naive_bayes(pos_train, neg_train, vocab, alpha=0)
    print('Evaluation without log')
    model1.evaluate(pos_test, neg_test, log=False)

    print('Evaluation with log')
    model1.evaluate(pos_test, neg_test, log=True)

def naive_bayes2():
    (pos_train, neg_train, vocab), (pos_test, neg_test) = load_data(0.2,0.2,0.2,0.2)
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]

    y_range = np.arange(0)
    x_range = np.arange(0)
    for alpha in alphas:
        print("alpha = ", alpha)
        model2 = naive_bayes(pos_train, neg_train, vocab, alpha=alpha)
        accuarcy = model2.evaluate(pos_test, neg_test, log=True)
        y_range = np.append(y_range,[accuarcy])
        x_range = np.append(x_range,[np.log(alpha)])

    for x, y in zip(x_range, y_range):
        label = "{:.3f}".format(y)

        plt.annotate(label,
                     (x, y),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.xlabel('Alpha values by log')
    plt.ylabel('Accuarcy')
    plt.plot(x_range,y_range,'bo-')
    plt.show()

def naive_bayes3():
    (pos_train, neg_train, vocab), (pos_test, neg_test) = load_data(1.0,1.0,1.0,1.0)
    alphas = [10]

    for alpha in alphas:
        print("Highest alpha = ", alpha)
        model3 = naive_bayes(pos_train, neg_train, vocab, alpha=alpha)
        model3.evaluate(pos_test, neg_test, log=True)




def naive_bayes4():
    (pos_train, neg_train, vocab), (pos_test, neg_test) = load_data(0.5,0.5,1.0,1.0)
    alphas = [10]
    for alpha in alphas:
        print("Highest alpha = ", alpha)
        model4 = naive_bayes(pos_train, neg_train, vocab, alpha=alpha)
        model4.evaluate(pos_test, neg_test, log=True)



def naive_bayes6():
    (pos_train, neg_train, vocab), (pos_test, neg_test) = load_data(0.1,0.5,1.0,1.0)
    alphas = [10]
    for alpha in alphas:
        print("Highest alpha = ", alpha)
        model6 = naive_bayes(pos_train, neg_train, vocab, alpha=alpha)
        model6.evaluate(pos_test, neg_test, log=True)


# naive_bayes1()
# naive_bayes2()
# naive_bayes3()
# naive_bayes4()
# naive_bayes6()