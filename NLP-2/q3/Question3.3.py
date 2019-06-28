import os
import torch.nn as nn
import string
from string import punctuation
from os import listdir
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

#################################################################################
#Reported Best Accuracy with Parameters-Question 3.3
#Activation Function : identity
#Optimizer/Solver Function : lbfgs
#First Hidden Layer Nodes : 48
#Best Accuracy : 0.7608445893494798
#################################################################################
model = KeyedVectors.load_word2vec_format(os.getcwd()+'\\GoogleNews-vectors-negative300.bin.gz', binary=True)
def main():
    path = os.getcwd() + "\\train\\pos\\"
    path1 = os.getcwd() + "\\train\\neg\\"
    data = []
    y = []
    for d in os.listdir(path):
        f = open(path + d, 'r')
        y.append(1)
        for line in f:
            data.append(line)

    for d in os.listdir(path1):
        f3 = open(path1 + d, 'r')
        y.append(-1)
        for line in f3:
            data.append(line)
    s = np.array(y)
    V1 = []
    
    for sentence in data:
      c=sent_vectorizer(word_tokenize(sentence), model)
      V1.append(c)
    V= np.array(V1)
    X_train,Y_train,X_test,Y_test=getDataForKFold(V,s,10)

    # activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
    # hidden_layer1_sizes={45,46,47,48,49,50}
    averageAccu=[]
    activation=[]
    hiddenLayer=[]
    act = ['identity','logistic','tanh', 'relu']
    f1 = open(os.getcwd() + "\\output\\q_3_3_result.txt", "w")
    for i in range(45,51):
            for k in act:
                average=[]
                # f1.write("Activation Function : " + str(k))
                # f1.write("\n")
                # f1.write("Optimizer/Solver Function : lbfgs" )
                # f1.write("\n")
                # f1.write("First Hidden Layer Nodes : " + str(i))
                # f1.write("\n")
                clf = MLPClassifier(activation=k, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, 10,),max_iter=30)

                for j in range(0, 10):
                    clf.fit(X_train[j], Y_train[j])
                    y_pred = clf.predict(X_test[j])
                    accuracy = accuracy_score(Y_test[j], y_pred)
                    average.append(accuracy)
                avgAccuracy=sum(average)/len(average)
                averageAccu.append(avgAccuracy)
                activation.append(k)
                hiddenLayer.append(i)
                # print(avgAccuracy)
                # f1.write("Average Accuracy : "+str(avgAccuracy))
                # f1.write("\n")
                # f1.write("\n")
    f1.write("Best Accuracy:  " + str(max(averageAccu)))
    indexOfMaxAccuracy = averageAccu.index(max(averageAccu))
    activ = activation[indexOfMaxAccuracy]
    hid = hiddenLayer[indexOfMaxAccuracy]
    f1.write("\n")
    f1.write("Optimal number of nodes in First Hidden Layer :"+str(hid))
    f1.write("\n")
    f1.write("Optimal Activation Function: "+ str(activ))
    f1.write("\n")
    f1.close()
##########################################################################################
# Retraining with the best parameters-Question 3.3
# Activation Function : identity
# Optimizer/Solver Function : lbfgs
# First Hidden Layer Nodes : 48
# Final Accuracy:  0.8075904677846425
##########################################################################################

    f2 = open(os.getcwd() + "\\output\\q_3_3_result.txt", "a+")
    clf1= MLPClassifier(activation=activ, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hid, 10,), max_iter=30)
    clf1.fit(V, s)
    y_pred = clf1.predict(V)
    accuracy = accuracy_score(s, y_pred)
    f2.write("\n")
    f2.write("\n")
    f2.write("Final Accuracy:  " + str(accuracy))
    f2.write("\n")
    f2.close()
###########################################################################################
#getting average word embeddings
###########################################################################################
def sent_vectorizer(sent, model):
  sent_vec = []
  numw = 0
  for w in sent:
    try:
      if numw == 0:
        sent_vec = model[w]
      else:
        sent_vec = np.add(sent_vec, model[w])
      numw += 1
    except:
      pass

  return np.divide(sent_vec,numw)
#######################################################################################
# Function for getting K Fold data for training and testing
#######################################################################################
def getDataForKFold(V, s, foldSize):
    rem = len(V) % 10
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    for fold in range(1,foldSize+1):
      if fold <= rem:
        if (fold == 1):
            test_index_start = int(len(V) / 10) * (fold - 1)
            test_index_end = int(test_index_start + (len(V) / 10))
        else:
            test_index_start = int(len(V) / 10) * (fold - 1) + (fold - 1)
            test_index_end = int(test_index_start + (len(V) / 10))
      else:
        test_index_start = int(len(V) / 10) * (fold - 1) + rem
        test_index_end = int(test_index_start + (len(V) / 10) - 1)
      X_test.append(np.array(V[slice(test_index_start, test_index_end + 1)]))
      Y_test.append(np.array(s[slice(test_index_start, test_index_end + 1)]))
      if (test_index_start == 0):
         X_train.append(np.array(V[slice(test_index_end + 1, len(V))]))
         Y_train.append(np.array(s[slice(test_index_end + 1, len(V))]))
      elif (test_index_end == len(V) - 1):
         X_train.append(np.array(V[slice(0, test_index_start)]))
         Y_train.append(np.array(s[slice(0, test_index_start)]))
      else:
         X_train.append(np.array(
            list(V[slice(0, test_index_start)]) + list(V[slice(test_index_end + 1, len(V))])))
         Y_train.append(np.array(list(s[slice(0, test_index_start)]) + list(s[slice(test_index_end + 1, len(V))])))

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__": main()