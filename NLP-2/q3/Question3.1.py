import torch.nn as nn
import string
from string import punctuation
import os
from os import listdir
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
import warnings

warnings.filterwarnings("ignore")


################################################################################
# Optimization Parameters: I have used 4 types of activation functions
# activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
# Activation function for the hidden layer.
#
# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
#  The solver/optimizer that I am using is lbfgs, shuffling is not required for this solver.
# I am using number of nodes in first hidden layer between 45 and 50 for optimization process.
# I have a total of 24 combinations of the parameters and every iteration will run 10 fold times.
# so there are 240 iterations in total. The highest average accuracy parameters
# for any out of 24 combinations
# are taken for retraining for 3.2.
# Reported Best Accuracy with Parameters-Question 3.1
# Activation Function : identity
# Optimizer/Solver Function : lbfgs
# First Hidden Layer Nodes : 47
# Best Accuracy : 0.7722403353516534
################################################################################
def main():
    path = os.getcwd() + "\\train\\pos\\"
    path1 = os.getcwd() + "\\train\\neg\\"
    data = []
    y = []
    text = ""
    for d in os.listdir(path):
        f = open(path + d, 'r')
        y.append(1)
        for line in f:
            data.append(line)
            text += line
    for d in os.listdir(path1):
        f3 = open(path1 + d, 'r')
        y.append(-1)
        for line in f3:
            data.append(line)
            text += line
    words = word_tokenize(text.lower())
    mwords = {}
    for w in words:
        mwords.setdefault(w, 0)
        mwords[w] += 1

    for i, w in enumerate(mwords, 0):
        mwords[w] = i
    feature = []
    sentFeature = []
    for i in range(len(mwords)):
        sentFeature.append(0)
    for d in data:
        sentFeature = []
        for i in range(len(mwords)):
            sentFeature.append(0)
        a = word_tokenize(d.lower())
        for j in a:
            x = mwords.get(j)
            sentFeature[x] += 1
        feature.append(sentFeature)
    newfeature = np.array(feature)
    s = np.array(y)
    X_train, Y_train, X_test, Y_test = getDataForKFold(newfeature, s, 10)
    # # activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
    # # hidden_layer1_sizes={45,46,47,48,49,50}
    averageAccu = []
    activation = []
    hiddenLayer = []
    act = ['identity', 'logistic', 'tanh', 'relu']
    f1 = open(os.getcwd() + "\\output\\q_3_1_result.txt", "w")
    for i in range(45, 51):
        for k in act:
            average = []
            f1.write("Activation Function : " + str(k))
            f1.write("\n")
            f1.write("Optimizer/Solver Function : lbfgs")
            f1.write("\n")
            f1.write("First Hidden Layer Nodes : " + str(i))
            f1.write("\n")
            clf = MLPClassifier(activation=k, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, 10,), max_iter=30)

            for j in range(0, 10):
                clf.fit(X_train[j], Y_train[j])
                y_pred = clf.predict(X_test[j])
                accuracy = accuracy_score(Y_test[j], y_pred)
                average.append(accuracy)
                print(accuracy)
            averageAccuracy = sum(average) / len(average)
            averageAccu.append(averageAccuracy)
            activation.append(k)
            hiddenLayer.append(i)
            print(averageAccuracy)
            f1.write("Average Accuracy : " + str(averageAccuracy))
            f1.write("\n")
            f1.write("\n")
    f1.write("Best Accuracy:  " + str(max(averageAccu)))
    indexOfMaxAccuracy = averageAccu.index(max(averageAccu))
    activ = activation[indexOfMaxAccuracy]
    hid = hiddenLayer[indexOfMaxAccuracy]
    f1.write("\n")
    f1.write("Optimal number of nodes in First Hidden Layer :" + str(hid))
    f1.write("\n")
    f1.write("Optimal Activation Function: " + str(activ))
    f1.write("\n")
    f1.close()
    ##########################################################################################
    # Retraining with the best parameters-Question 3.2
    # Activation Function : identity
    # Optimizer/Solver Function : lbfgs
    # First Hidden Layer Nodes : 47
    # Final Accuracy: 0.9488084730803178
    ##########################################################################################
    f2 = open(os.getcwd() + "\\output\\q_3_2_result.txt", "w")
    clf1 = MLPClassifier(activation=activ, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hid, 10,), max_iter=30)
    clf1.fit(newfeature, s)
    y_pred = clf1.predict(newfeature)
    accuracy = accuracy_score(s, y_pred)
    f2.write("Final Accuracy:  " + str(accuracy))
    f2.write("\n")
    f2.close()


#######################################################################################
# Function for getting K Fold data for training and testing
#######################################################################################
def getDataForKFold(newfeature, s, foldSize):
    rem = len(newfeature) % 10
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for fold in range(1, foldSize + 1):
        if fold <= rem:
            if (fold == 1):
                test_index_start = int(len(newfeature) / 10) * (fold - 1)
                test_index_end = int(test_index_start + (len(newfeature) / 10))
            else:
                test_index_start = int(len(newfeature) / 10) * (fold - 1) + (fold - 1)
                test_index_end = int(test_index_start + (len(newfeature) / 10))
        else:
            test_index_start = int(len(newfeature) / 10) * (fold - 1) + rem
            test_index_end = int(test_index_start + (len(newfeature) / 10) - 1)
        X_test.append(np.array(newfeature[slice(test_index_start, test_index_end + 1)]))
        Y_test.append(np.array(s[slice(test_index_start, test_index_end + 1)]))
        if (test_index_start == 0):
            X_train.append(np.array(newfeature[slice(test_index_end + 1, len(newfeature))]))
            Y_train.append(np.array(s[slice(test_index_end + 1, len(newfeature))]))
        elif (test_index_end == len(newfeature) - 1):
            X_train.append(np.array(newfeature[slice(0, test_index_start)]))
            Y_train.append(np.array(s[slice(0, test_index_start)]))
        else:
            X_train.append(np.array(
                list(newfeature[slice(0, test_index_start)]) + list(
                    newfeature[slice(test_index_end + 1, len(newfeature))])))
            Y_train.append(
                np.array(list(s[slice(0, test_index_start)]) + list(s[slice(test_index_end + 1, len(newfeature))])))

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__": main()
