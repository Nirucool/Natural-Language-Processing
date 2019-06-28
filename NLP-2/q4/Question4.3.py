import torch.nn as nn
import string
from string import punctuation
import os
from os import listdir
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from scipy.stats import pearsonr
from big_phoney import BigPhoney
import re
import warnings
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import readability

warnings.filterwarnings("ignore")

nltk.download('stopwords')


def main():
    path = os.getcwd() + "\\train\\"
    trainSummary = []
    trainFluency = []
    trainNRedundancy = []
    stops = set(stopwords.words("english"))
    ##############################################################
    # reading and pre processing training data
    ##############################################################
    with open(path + 'Train_Data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in readCSV:
            ####first row -headings not required#####
            if (count == 0):
                count += 1
                continue
            ########for rows with no data############
            if (row[0] == ""):
                continue
            ###########################################################
            # replace \n and \t with spaces
            # remove extra spaces
            # convert to lowercase
            ###########################################################
            a = row[0]
            a = a.replace("\n", " ")
            a = a.replace("\t", " ")
            a = a.lower()
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

    testSummary = []
    testFluency = []
    testNRedundancy = []
    with open(path + 'Test_Data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in readCSV:
            if (count == 0):
                count += 1
                continue
            if (row[0] == ""):
                continue
            a = row[0]
            a = a.replace("\n", " ")
            a = a.replace("\t", " ")
            a = a.lower()
            a = re.sub(" +", " ", a)
            testSummary.append(a)
            ######################################################################
            # changing fluency and redundancy values from string to float or int
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
    # Total number of repetitive unigrams
    #####################################################################
    unigram_feature = []
    for sen in trainSummary:
        sen = word_tokenize(sen)
        unigram = {}
        words = []

        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words)):
            t = "_".join(words[j:j + 1])
            unigram.setdefault(t, 0)
            unigram[t] += 1
        count = 0
        for k in unigram.values():
            if (k > 1):
                count += 1
        unigram_feature.append(count)

    testunigram_feature = []
    for sen in testSummary:
        sen = word_tokenize(sen)
        unigram1 = {}
        words = []
        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words)):
            t = "_".join(words[j:j + 1])
            unigram1.setdefault(t, 0)
            unigram1[t] += 1
        count1 = 0
        for k in unigram1.values():
            if (k > 1):
                count1 += 1
        testunigram_feature.append(count1)

    ##################################################################
    # Total number of repetitive bigrams
    ###################################################################
    bigram_feature = []
    for sen in trainSummary:
        sen = word_tokenize(sen)
        bigram = {}
        words = []
        for j in sen:
            if j not in stops:
                words.append(j)
        for j in range(len(words) - 1):
            t = "_".join(words[j:j + 2])
            bigram.setdefault(t, 0)
            bigram[t] += 1
        count = 0
        for k in bigram.values():
            if (k > 1):
                count += 1
        bigram_feature.append(count)

    testbigram_feature = []
    for sen in testSummary:
        sen = word_tokenize(sen)
        bigram1 = {}
        words1 = []
        for j in sen:
            if j not in stops:
                words1.append(j)
        for j in range(len(words1) - 1):
            t = "_".join(words1[j:j + 2])
            bigram1.setdefault(t, 0)
            bigram1[t] += 1
        count1 = 0
        for k in bigram1.values():
            if (k > 1):
                count1 += 1
        testbigram_feature.append(count1)

    ########################################################################
    # Minimum Flesch reading-ease score:
    ########################################################################

    flesch = []
    phoney = BigPhoney()
    for j in trainSummary:
        min = float('inf')
        sen = nltk.tokenize.sent_tokenize(j)
        for k in sen:
            words = word_tokenize(k)
            count = 0
            for z in words:
                count += phoney.count_syllables(z)
            score = readability.FleschReadingEase(count, len(words), 1)
            if (score < min):
                min = score
        flesch.append(min)

    fleschT = []

    for j in testSummary:
        min = float('inf')
        sen = nltk.tokenize.sent_tokenize(j)
        read = []

        for k in sen:
            words = word_tokenize(k)
            count = 0

            for z in words:
                count += phoney.count_syllables(z)
            score = readability.FleschReadingEase(count, len(words), 1)
            if (score < min):
                min = score
        fleschT.append(min)
    ##############################################################################################
    # Classifier for the above three features-Q4.3
    # Linear Regression Model

    ###############################################################################################
    unigramTrain = np.array(unigram_feature).reshape(len(unigram_feature), 1)
    unigramTest = np.array(testunigram_feature).reshape(len(testunigram_feature), 1)
    bigramTrain = np.array(bigram_feature).reshape(len(bigram_feature), 1)
    bigramTest = np.array(testbigram_feature).reshape(len(testbigram_feature), 1)
    fleschTrain = np.array(flesch).reshape(len(flesch), 1)
    fleschTest = np.array(fleschT).reshape(len(fleschT), 1)
    clf = LinearRegression()
    #######################################################################################
    # Reported Values
    # scipy.stats.pearsonr(x, y) gives two values, first value gives the value -1 to 1 with
    # positive value referring to more correlation.
    # The second p-value roughly indicates the probability of an uncorrelated system producing datasets that
    # have a Pearson correlation at least as extreme as the one computed from these datasets.
    # The p-values are not entirely reliable but are probably reasonable for datasets larger than 500 or so.
    # MSE - 0.22993706600411773
    # Pearson Correlation Coefficient-(0.3521331207011803, 3.163607855250048e-07)
    #######################################################################################
    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), fleschTrain)), np.array(trainFluency))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), fleschTest)))
    MSE = mean_squared_error(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(pearSon)

###############################################################################################
    # Question 4.4
    # feature 1
    # Maximum value of SMOG index,a Simple Measure of Gobbledygook
    # The value of SMOG index ranges from 1 to 240 with the higher value for less readable or
    # less fluent. It uses words with more than 3 syllables to determine complexity of the sentence.
    # This gives a measure of the fluency or readability or understandability of the summaries
    # and hence reduces the MSE and increases Pearson Correlation Coefficient
###############################################################################################

    grade = []
    for j in trainSummary:
        max = float('-inf')
        sen = nltk.tokenize.sent_tokenize(j)

        for k in sen:
            words = word_tokenize(k)
            count = 0
            for z in words:
                c = phoney.count_syllables(z)
                if (c >= 3):
                    count += 1
            score = readability.SMOGIndex(count, 1)
            if (score > max):
                max = score
        grade.append(max)

    gradeT = []
    for j in testSummary:
        max = float('-inf')
        sen = nltk.tokenize.sent_tokenize(j)

        for k in sen:
            words = word_tokenize(k)
            count = 0
            for z in words:
                c = phoney.count_syllables(z)
                if (c >= 3):
                    count += 1
            score = readability.SMOGIndex(count, 1)
            if (score > max):
                max = score
        gradeT.append(max)
###############################################################################################
# Question 4.4
# feature 2
# Lix Readability Formula
# LIX = A/B + (C x 100)/A, where

#A = Number of words
#B = Number of periods (defined by period, colon or capital first letter)
#C = Number of long words (More than 6 letters)

# LIX uses words with more than six letters to determine the complexity of the sentence.
# More is the LIX score, more is the complexity and less is the fluency. Hence LIX score gives
# a fair measure of readability and decreases MSE and increases Pearson Correlation Coefficient.
###############################################################################################
    lix = []
    for j in trainSummary:
        sen = nltk.tokenize.sent_tokenize(j)
        words = word_tokenize(j)
        count = 0
        for z in words:
            if (len(z) > 6):
                count += 1
        score = readability.LIX(len(words), count, len(sen))
        lix.append(score)

    lixT = []

    for j in testSummary:
        sen = nltk.tokenize.sent_tokenize(j)
        words = word_tokenize(j)
        count = 0
        for z in words:
            if (len(z) > 6):
                count += 1
        score = readability.LIX(len(words), count, len(sen))

        lixT.append(score)

    ####################################################################################

    gradeTrain = np.array(grade).reshape(len(grade), 1)
    gradeTest = np.array(gradeT).reshape(len(gradeT), 1)
    lixTrain = np.array(lix).reshape(len(lix), 1)
    lixTest = np.array(lixT).reshape(len(lixT), 1)
    ##############################################################################################
    # Classifier 2-Same as above using Linear Regression Model-Q4.4
    # Using one additional feature of Maximum value of SMOG Index
    # Reported Values:
    # MSE-0.22948742171238876
    # Pearson Correlation Coefficient-(0.3547109953705656, 2.5568576898239954e-07)
    ##############################################################################################

    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), np.hstack((fleschTrain, gradeTrain)))),
            np.array(trainFluency))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), np.hstack((fleschTest, gradeTest)))))
    MSE = mean_squared_error(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(pearSon)

    #######################################################################################################
    # Classifier 3-Same as above using Linear Regression Model-Q4.4
    # Using one additional feature of LIX Readability formula
    # Reported Values:
    # MSE-0.22856596250744246
    # Pearson Correlation Coefficient-(0.3545475618365523, 2.591754092042154e-07)
    #######################################################################################################
    clf.fit(np.hstack((np.hstack((unigramTrain, bigramTrain)), np.hstack((fleschTrain, lixTrain)))),
            np.array(trainFluency))
    y_pred = clf.predict(np.hstack((np.hstack((unigramTest, bigramTest)), np.hstack((fleschTest, lixTest)))))
    MSE = mean_squared_error(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(MSE)
    pearSon = pearsonr(np.float64(np.array(testFluency)), np.float64(np.array(y_pred)))
    print(pearSon)



if __name__ == "__main__": main()
