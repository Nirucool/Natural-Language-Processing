import torch
import string
from string import punctuation
import os
from os import listdir
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from numpy.linalg import norm
from gensim.models import KeyedVectors
from scipy.stats import pearsonr
import re
import warnings
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import pairwise_kernels
warnings.filterwarnings("ignore")

model = KeyedVectors.load_word2vec_format(os.getcwd()+'\\GoogleNews-vectors-negative300.bin.gz', binary=True)

nltk.download('stopwords')

###############################################################
#driver method main
###############################################################
def main():
    path = os.getcwd() + "\\train\\"
    trainSummary=[]
    trainFluency=[]
    trainNRedundancy=[]
    stops = set(stopwords.words("english"))
    ##############################################################
    # reading and pre processing training data
    ##############################################################
    with open(path+'Train_Data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count=0
        for row in readCSV:
            ####first row -headings not required#####
            if(count==0):
                count+=1
                continue
            ########for rows with no data############
            if(row[0]==""):
                continue
            ###########################################################
            # replace \n and \t with spaces
            # remove extra spaces
            # convert to lowercase
            ###########################################################
            a=row[0]
            a=a.replace("\n", " ")
            a=a.replace("\t", " ")
            a=a.lower()
            a = re.sub(" +", " ", a)
            trainSummary.append(a)
            #####redundancy and fluency values are strings-conversion to int and float######
            if (str(row[1]).__contains__(".")):
                x = float(str(row[1]))
                trainNRedundancy.append(x)
            else:
                x = int(str(row[1]))
                trainNRedundancy.append(x)

            if (str(row[2]).__contains__(".")):
                x = float(str(row[2]))
                trainFluency.append(x)
            else:
                x = int(str(row[2]))
                trainFluency.append(x)
    ##############################################################
    # reading and pre processing test data
    ##############################################################

    testSummary=[]
    testFluency=[]
    testNRedundancy=[]
    with open(path+'Test_Data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count=0
        for row in readCSV:
            if (count == 0):
                count += 1
                continue
            if (row[0] == ""):
                continue
            a=row[0]
            a=a.replace("\n", " ")
            a=a.replace("\t", " ")
            a=a.lower()
            a = re.sub(" +", " ", a)
            testSummary.append(a)
            ######################################################################
            #changing fluency and redundancy values from string to float or int
            ######################################################################
            if (str(row[1]).__contains__(".")):
                x = float(str(row[1]))
                testNRedundancy.append(x)
            else:
                x = int(str(row[1]))
                testNRedundancy.append(x)

            if (str(row[2]).__contains__(".")):
                x = float(str(row[2]))
                testFluency.append(x)
            else:
                x = int(str(row[2]))
                testFluency.append(x)


#####################################################################
    # Maximum repetition of unigrams:
#####################################################################
    unigram_feature=[]
    for sen in trainSummary:
        sen=word_tokenize(sen)
        unigram={}
        max=0
        words=[]

        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words)):
            t = "_".join(words[j:j + 1])
            unigram.setdefault(t, 0)
            unigram[t] += 1
        for k in unigram.values():
            if(k>max):
                max=k
        unigram_feature.append(max)

    testunigram_feature = []
    for sen in testSummary:
        sen = word_tokenize(sen)
        unigram1 = {}
        max = 0
        words=[]
        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words)):
            t = "_".join(words[j:j + 1])
            unigram1.setdefault(t, 0)
            unigram1[t] += 1
        for k in unigram1.values():
            if (k > max):
                max = k
        testunigram_feature.append(max)

#####################################################################
    # Maximum repetition of bigrams
#####################################################################
    bigram_feature = []
    for sen in trainSummary:
        sen = word_tokenize(sen)
        bigram = {}
        max = 0
        words=[]
        for j in sen:
           if j not in stops:
               words.append(j)
        for j in range(len(words)-1):
            t="_".join(words[j:j+2])
            bigram.setdefault(t, 0)
            bigram[t] += 1
        for k in bigram.values():
            if (k > max):
                max = k
        bigram_feature.append(max)

    testbigram_feature = []
    for sen in testSummary:
        sen = word_tokenize(sen)
        bigram1 = {}
        max = 0
        words1 = []
        for j in sen:
            if j not in stops:
                words1.append(j)
        for j in range(len(words1) - 1):
            t = "_".join(words1[j:j + 2])
            bigram1.setdefault(t, 0)
            bigram1[t] += 1
        for k in bigram1.values():
            if (k > max):
                max = k
        testbigram_feature.append(max)


############################################################################
    #Maximum Sentence Similarity-cosine similarity
############################################################################

    cos = []
    for j in range(len(trainSummary)):
        arr = []
        sent = nltk.tokenize.sent_tokenize(trainSummary[j])
        sen=[]
        for z in sent:
            s = z.translate(str.maketrans('', '', string.punctuation))
            if(len(words)>0):
                sen.append(s)
        for k in range(len(sen)):
            b=sent_vectorizer(word_tokenize(sen[k]), model)
            if(len(b)>0):
             arr.append(sent_vectorizer(word_tokenize(sen[k]), model))

        Vk= np.array(arr)
        a=cosine_similarity(Vk)
        max = float('-inf')
        for l,z in enumerate(a,0):
            for m,n in enumerate(z,0):
              if l != m and n > max:
                max = n
        #####if only one sentence in summary#########################
        if(max==float("-inf")):
          max=0.0
        cos.append(max)

    cosT = []
    for j in range(len(testSummary)):
        arr = []
        sent = nltk.tokenize.sent_tokenize(testSummary[j])
        sen=[]
        for z in sent:
            s = z.translate(str.maketrans('', '', string.punctuation))
            if(len(words)>0):
                sen.append(s)
        for k in range(len(sen)):
            b=sent_vectorizer(word_tokenize(sen[k]), model)
            if(len(b)>0):
             arr.append(sent_vectorizer(word_tokenize(sen[k]), model))

        Vk= np.array(arr)
        a=cosine_similarity(Vk)
        max = float('-inf')
        for l, z in enumerate(a, 0):
            for m, n in enumerate(z, 0):
                if l != m and n > max:
                    max = n
        #########if only one sentence in summary####################
        if (max == float("-inf")):
            max = 0.0
        cosT.append(max)


###############################################################################################
    #Classifier for the above three features
    # Linear Regression Model

###############################################################################################
    unigramTrain = np.array(unigram_feature).reshape(len(unigram_feature), 1)
    unigramTest = np.array(testunigram_feature).reshape(len(testunigram_feature), 1)
    bigramTrain = np.array(bigram_feature).reshape(len(bigram_feature), 1)
    bigramTest = np.array(testbigram_feature).reshape(len(testbigram_feature), 1)
    cosTrain = np.array(cos).reshape(len(cos), 1)
    cosTest = np.array(cosT).reshape(len(cosT), 1)
    clf = LinearRegression()
    #######################################################################################
    # Reported Values
    # scipy.stats.pearsonr(x, y) gives two values, first value gives the value -1 to 1 with
    # positive value referring to more correlation.
    # The second p-value roughly indicates the probability of an uncorrelated system producing datasets that
    # have a Pearson correlation at least as extreme as the one computed from these datasets.
    # The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
    # MSE - 0.22461729023458354
    # Pearson Correlation Coefficient-(0.673167066845756, 9.354069464157302e-28)
    #######################################################################################
    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), cosTrain)), np.array(trainNRedundancy))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), cosTest)))
    MSE = mean_squared_error(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(pearSon)
################################################################################################
    # Question 4.2
    # feature 1
    # PAIRWISE KERNELS - Kernels are measures of similarity, i.e. s(a, b) > s(a, c) if objects
    # a and b are considered “more similar” than objects a and c. A kernel must also be positive
    # semi-definite.
    # Here, the pairwise kernel similarity is found using word2vec vectors between all sentences
    # in a summary. The max value is taken into account as feature for calculating MSE and Pearson Correlation
    # Coefficient. More the similarity, more is the redundancy. Hence it improves MSE and Pearson
    # correlation coefficient.

################################################################################################
    kernel = []
    for j in range(len(trainSummary)):
        arr = []
        sent = nltk.tokenize.sent_tokenize(trainSummary[j])
        sen = []
        for z in sent:
            s = z.translate(str.maketrans('', '', string.punctuation))
            if (len(word_tokenize(s)) > 0):
                sen.append(s)
        for k in range(len(sen)):
            b = sent_vectorizer(word_tokenize(sen[k]), model)
            if (len(b) > 0):
                arr.append(sent_vectorizer(word_tokenize(sen[k]), model))

        Vk = np.array(arr)
        a=pairwise_kernels(Vk)
        max = float('-inf')
        for l, z in enumerate(a, 0):
            for m, n in enumerate(z, 0):
                if l != m and n > max:
                    max = n
        if (max == float("-inf")):
            max = 0.0
        kernel.append(max)

    kernelT = []
    for j in range(len(testSummary)):
        arr1 = []
        sent = nltk.tokenize.sent_tokenize(testSummary[j])
        sen = []
        for z in sent:
            s = z.translate(str.maketrans('', '', string.punctuation))
            if (len(word_tokenize(s)) > 0):
                sen.append(s)
        for k in range(len(sen)):
            b = sent_vectorizer(word_tokenize(sen[k]), model)
            if (len(b) > 0):
                arr1.append(sent_vectorizer(word_tokenize(sen[k]), model))

        Vk = np.array(arr1)
        a = pairwise_kernels(Vk)
        max = float('-inf')
        for l, z in enumerate(a, 0):
            for m, n in enumerate(z, 0):
                if l != m and n > max:
                    max = n
        if (max == float("-inf")):
            max = 0.0
        kernelT.append(max)

######################################################################################
    # Maximum repetition of trigrams
    # More is the repetition of trigrams, more will the repetition count which reflects
    # redundancy and less count reflects non redundancy. Hence it improves MSE and Pearson
    # correlation coefficient values for redundancy.
#######################################################################################
    trigram_feature = []
    for sen in trainSummary:
        sen = word_tokenize(sen)
        trigram = {},
        max = 0
        words = []
        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words) - 2):
            t = "_".join(words[j:j + 3])
            trigram.setdefault(t, 0)
            trigram[t] += 1
        for k in trigram.values():
            if (k > max):
                max = k
        trigram_feature.append(max)

    testtrigram_feature = []
    for sen in testSummary:
        sen = word_tokenize(sen)
        trigram1 = {}
        max = 0
        words1 = []
        for j in sen:
            if j not in stops:
                words1.append(j)
        for j in range(len(words1) - 2):
            t = "_".join(words1[j:j + 3])
            trigram1.setdefault(t, 0)
            trigram1[t] += 1
        for k in trigram1.values():
            if (k > max):
                max = k
        testtrigram_feature.append(max)

    trigramTrain = np.array(trigram_feature).reshape(len(trigram_feature), 1)
    trigramTest = np.array(testtrigram_feature).reshape(len(testtrigram_feature), 1)
    kernelTrain = np.array(kernel).reshape(len(kernel), 1)
    kernelTest = np.array(kernelT).reshape(len(kernelT), 1)

#########################################################################################
    # Classifier 2-Same as above using Linear Regression Model-Q4.2
    # Using one additional feature of Pairwise Kernel Similarity
    # Reported Values:
    # MSE-0.21585679358762774
    # Pearson Correlation Coefficient-(0.6944329487211355, 4.0559043109817306e-30)
#########################################################################################
    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), np.hstack((cosTrain, kernelTrain)))), np.array(trainNRedundancy))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), np.hstack((cosTest, kernelTest)))))
    MSE = mean_squared_error(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(pearSon)

#########################################################################################
    # Classifier 3-Same as above using Linear Regression Model-Q4.2
    # Using one additional feature of Maximum repetition of trigrams
    # Reported Values:
    # MSE-0.21894478133177522
    # Pearson Correlation Coefficient-(0.6775029132194252, 3.2014277319324223e-28)
#########################################################################################

    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), np.hstack((cosTrain, trigramTrain)))),
            np.array(trainNRedundancy))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), np.hstack((cosTest, trigramTest)))))
    MSE = mean_squared_error(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testNRedundancy)), np.float64(np.array(y_pred)))
    print(pearSon)

#######################word2vec average calculation########################################
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