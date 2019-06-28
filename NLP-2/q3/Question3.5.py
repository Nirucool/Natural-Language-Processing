import torch.nn as nn
import string
from string import punctuation
import os
from os import listdir
import numpy as np
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from nltk import pos_tag,word_tokenize
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

model = KeyedVectors.load_word2vec_format(os.getcwd()+'\\GoogleNews-vectors-negative300.bin.gz', binary=True)
################################################################################
# Reported Best Accuracy with Parameters-Used from Question 3.4 model
# Activation Function : identity
# Optimizer/Solver Function : lbfgs
# First Hidden Layer Nodes : 48
################################################################################
def main():
    path = os.getcwd() + "\\train\\pos\\"
    path1 = os.getcwd() + "\\train\\neg\\"
    pathTest = os.getcwd() + "\\test\\"

    data = []
    y = []
    text = ""
    countOfTraining=0
    countOfTest=0
    # adding positive files
    for d in os.listdir(path):
        y.append(1)
        countOfTraining+=1
        f = open(path + d, 'r')
        for line in f:
            data.append(line)
            text += line
        f.close()
    # adding negative files
    for d in os.listdir(path1):
        y.append(-1)
        countOfTraining+=1
        f3 = open(path1 + d, 'r')
        for line in f3:
            data.append(line)
            text += line
        f3.close()
    #adding test files
    for d in os.listdir(pathTest):
        countOfTest+=1
        f4 = open(pathTest + d, 'r')
        for line in f4:
            data.append(line)
            text += line
        f4.close()

    feature = []
    testFeature=[]
    sentFeature = []
    word = pos_tag(word_tokenize(text.lower()))
    words = []
    for j in word:
        t = '_'.join(j)
        words.append(t)
    mwords = {}
    for w in words:
        mwords.setdefault(w, 0)
        mwords[w] += 1

    for i, w in enumerate(mwords, 0):
        mwords[w] = i

    for d in data:
        sentFeature = []
        for i in range(len(mwords)):
            sentFeature.append(0)
        a = pos_tag(word_tokenize(d.lower()))
        k = []
        for l in a:
            t = '_'.join(l)
            k.append(t)
        for j in k:
            if j in mwords:
                x = mwords.get(j)
                sentFeature[x] += 1
        if (len(feature) < countOfTraining):
            feature.append(sentFeature)
        else:
            testFeature.append(sentFeature)
    newfeature = np.array(feature)
    newfeature1 = np.array(testFeature)
    V1 = []
    V2 = []
    for sentence in data:
        if (len(V1) < countOfTraining):
           V1.append(sent_vectorizer(word_tokenize(sentence), model))
        else:
           V2.append(sent_vectorizer(word_tokenize(sentence), model))
    V = np.array(V1)
    VT=np.array(V2)
    s = np.array(y)
    trainData = np.hstack((newfeature, V))
    testData = np.hstack((newfeature1,VT))

    files=[]
    for d in os.listdir(pathTest):
        files.append(d)


    ##########################################################################################
    # Retraining with the best parameters(from 3.4)-Question 3.5
    # Activation Function : identity
    # Optimizer/Solver Function : lbfgs
    # First Hidden Layer Nodes : 48
    ##########################################################################################
    clf1 = MLPClassifier(activation='identity', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(48, 10,), max_iter=30)
    clf1.fit(trainData, s)
    y_pred = clf1.predict(testData)
    f1 = open(os.getcwd() + "\\output\\pos.txt", "w")
    for i in range(len(y_pred)):
        if(y_pred[i]==1):
            f1.write(str(files[i]))
            f1.write("\n")
    f1.close()

    f2 = open(os.getcwd() + "\\output\\neg.txt", "w")
    for i in range(len(y_pred)):
        if (y_pred[i] == -1):
            f2.write(str(files[i]))
            f2.write("\n")
    f2.close()

#############################################################################################
# method to get average word embeddings
#############################################################################################
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


if __name__ == "__main__": main()