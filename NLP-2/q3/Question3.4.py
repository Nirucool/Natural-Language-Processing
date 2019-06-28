import os
import torch.nn as nn
import string
from string import punctuation
from os import listdir
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from nltk import pos_tag,word_tokenize
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import warnings

warnings.filterwarnings("ignore")

model = KeyedVectors.load_word2vec_format(os.getcwd()+'\\GoogleNews-vectors-negative300.bin.gz', binary=True)
def main():
    path = os.getcwd() + "\\train\\pos\\"
    path1 = os.getcwd() + "\\train\\neg\\"
    data = []
    y = []
    text=""
    for d in os.listdir(path):
        f = open(path + d, 'r')
        y.append(1)
        for line in f:
            data.append(line)
            text+=line
    for d in os.listdir(path1):
        f3 = open(path1 + d, 'r')
        y.append(-1)
        for line in f3:
            data.append(line)
            text+=line

    word= pos_tag(word_tokenize(text.lower()))
    words=[]
    for j in word:
        t='_'.join(j)
        words.append(t)
    mwords = {}
    for w in words:
        mwords.setdefault(w, 0)
        mwords[w] += 1

    for i, w in enumerate(mwords, 0):
        mwords[w] = i
    feature = []
    for d in data:
        sentFeature = []
        for i in range(len(mwords)):
            sentFeature.append(0)
        a = pos_tag(word_tokenize(d.lower()))
        k=[]
        for l in a:
            t = '_'.join(l)
            k.append(t)
        for j in k:
          if j in mwords:
            x = mwords.get(j)
            sentFeature[x] += 1
        feature.append(sentFeature)
    newfeature = np.array(feature)
    V1 = []
    for sentence in data:
        V1.append(sent_vectorizer(word_tokenize(sentence), model))
    V = np.array(V1)
    s = np.array(y)
    trainData=np.hstack((newfeature,V))
##########################################################################################
# Retraining with the best parameters from Question 3.3
# Activation Function : identity
# Optimizer/Solver Function : lbfgs
# First Hidden Layer Nodes : 48
# Final Accuracy: 0.9673433362753752
#############################################################################################

    f2 = open(os.getcwd() + "\\output\\q_3_4_result.txt", "w")
    clf1= MLPClassifier(activation='identity', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(48, 10,), max_iter=30)
    clf1.fit(trainData,s)
    y_pred = clf1.predict(trainData)
    accuracy = accuracy_score(s, y_pred)
    print(accuracy)
    f2.write("Final Accuracy:  " + str(accuracy))
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