import os
import re
import math
from decimal import Decimal, getcontext
import string
import collections
# used just for sentence extraction
from nltk import data
from nltk import tokenize
import nltk
nltk.download('punkt')



perplexity_trigram=[]
dictPerpFileName={}
perplexity_leverage=[]
lambda1=[]
lambda2=[]
lambda3=[]

###################################################################################
# main method/driver method#-Q3.2
# keep training data folder gutenberg in current working directory,
# remove Readme file present in it.
####################################################################################
def main():
   path = os.getcwd() + "\\gutenberg\\"
   # grid search...divide data in 80% training set, 20% heldout set, clean both sets and get ngrams from
   # 80% training set
   heldData, unigram, bigram, trigram = getNGramCountEightyPercentTrainingData(path)
   # get probability for 20% heldout set and find lambdas using grid search
   lam1,lam2,lam3=getLambdas(unigram, bigram, trigram, heldData)
   #prints lambdas for unigram, bigram and trigram respectively for linear interpolation
   print(lam1)
   print(lam2)
   print(lam3)
   print(perplexity_leverage)
   # process training files
   text=processAllTrainingFiles(path)
   # get unigrams, bigrams and trigrams from full training data(all files)
   trainUnigram, trainBigram, trainTrigram=findNGramCounts(text)

   #process test data
   data=processTestData()
   # get probabilities for test data
   getProbabilityTestData(trainUnigram,trainBigram,trainTrigram,data,lam1,lam2,lam3)
   # write all perplexities to PerplexitiesInterpolation.txt
   writePerplexityToFile(perplexity_trigram)
   # write top 50 perplexities for french files
   topFiftyPerplexities(dictPerpFileName,perplexity_trigram)

##########################################################################################
#getting unigram,bigram and trigram from training set ->80% of total number of sentences
##########################################################################################
def getNGramCountEightyPercentTrainingData(path):
    trigram = {}
    bigram = {}
    unigram = {}
    text=""
    #getting entire data to get sentences for training and heldout set
    for data in os.listdir(path):
        file = open(path + data, 'r')
        text = text + file.read()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence=(tokenize.sent_tokenize(text))
    # 80% training data sentence count
    size=int(len(sentence)*0.80)
    #heldOut data
    heldData=""
    for j in range(size, len(sentence)):
        heldData=heldData+sentence[j]

    #80percent training data
    dat=""
    for i in range(size):
       dat=dat+sentence[i]
    #dat = dat.rstrip()
    dat = dat.replace('\n', ' ')
    dat=dat.translate(str.maketrans('', '', string.punctuation))
    dat = re.sub(" +", " ", dat)
    d = {}
    for i in range(len(dat)):
        d.setdefault(dat[i], 0)
        d[dat[i]] += 1
    for num in d.keys():
        if d[num] <= 5:
            # Here I am using '#' instead of UNK to avoid U|N, N|K,
            # '#' is not present in training and test data so chose it
            dat = dat.replace(str(num), "#")
    #getting unigram, bigram and trigram for 80% training data
    for x in range(len(dat) - 3 + 1):
        t = ''.join(dat[x:x + 3])
        trigram.setdefault(t, 0)
        trigram[t] += 1

    for y in range(len(dat) - 2 + 1):
        b = ''.join(dat[y:y + 2])
        bigram.setdefault(b, 0)
        bigram[b] += 1

    for z in range(len(dat)):
        u = ''.join(dat[z:z + 1])
        unigram.setdefault(u, 0)
        unigram[u] += 1
    return heldData,unigram,bigram,trigram

##################################################################################
#get probabilities for heldout set -20% of training data
##################################################################################
def getLambdas(unigram,bigram,trigram,heldData):
    prob1=float(0.0)
    prob2=float(0.0)
    prob=float(0.0)
    #process held out data
    #heldData = heldData.rstrip()
    heldData = heldData.replace('\n', ' ')
    heldData = heldData.translate(str.maketrans('', '', string.punctuation))
    heldData = re.sub(" +", " ", heldData)
    d = {}
    for i in range(len(heldData)):
        d.setdefault(heldData[i], 0)
        d[heldData[i]] += 1
    for num in d.keys():
        if d[num] <= 5:
            # Here I am using '#' instead of UNK to avoid U|N, N|K,
            # '#' is not present in training and test data so chose it
            heldData = heldData.replace(str(num), "#")
    #################################################################################
    # get probabilities for held out set and find lambdas to maximize probabilities
    #################################################################################
    for x in range(len(heldData) - 3 + 1):
        t = ''.join(heldData[x:x + 3])
        if (t in trigram.keys()):
           x = trigram[t]
        else:
           x = 0
        if t[0:2] in bigram.keys():
            prob2=x/(bigram[t[0:2]])
        else:prob2=0
        if (t[0:2] in bigram.keys()):
                x = bigram[t[0:2]]
        else:x = 0
        if t[0:1] in unigram.keys():
            prob1=x/unigram[t[0:1]]
            prob = unigram[t[0:1]] / sum(unigram.values())
        else:
            prob1=0
            prob=unigram["#"]/sum(unigram.values())
        gridSearch(prob, prob1, prob2)
    # after getting all possible values of lambdas which gives high probabilities for held out set data,
    # find which combination gives lowest perplexity for heldout set
    # Since the model with low perplexity is a better model, hence we will find lambda values
    # which leads to lowest perplexity
    lam1,lam2,lam3=leveragingHeldOutPerplexity(unigram,bigram,trigram,lambda1,lambda2,lambda3,heldData)
    # perplexities calculated using above method gives 4 perplexities
    # [6.926007731845687, 10.604501833025008, 9.926656747330513, 13.13688127120477]
    # we chose lambda combination 0.1,0.1,0.8 which gave lowest perplexity 6.926007731845687
    return lam1,lam2,lam3
########################################################################
#use grid search to maximize probabilities
########################################################################
def gridSearch(prob,prob1,prob2):
      gridSearchValues=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
      finalProb=0.0
      lamb1=0.0
      lamb2=0.0
      lamb3=0.0

      for i in gridSearchValues:
        l1=i;
        a=(int)((float)(1.0-l1)*(float)(10))
        for j in range(a):
          l2=round(1.0-(l1+(float(j)/10)),1)
          l3=round(1.0-(l1+l2),1)
          a=prob*l1+prob1*l2+prob2*l3
          if(a>finalProb):
            finalProb=a
            lamb1=l1
            lamb2=l2
            lamb3=l3
      lambda1.append(lamb1)
      lambda2.append(lamb2)
      lambda3.append(lamb3)

########################################################################################
#getting lambdas leveraging perplexities for held out set
#########################################################################################
def leveragingHeldOutPerplexity(unigram,bigram,trigram,lambda1,lambda2,lambda3,heldData):
    perplex=float("inf")
    l1 = dict(collections.Counter(lambda1))
    l2 = dict(collections.Counter(lambda2))
    l3 = dict(collections.Counter(lambda3))
    for a in l1.keys():
        for b in l2.keys():
            for c in l3.keys():
                if(a+b+c)==1.0:
                    probability=[]
                    for x in range(len(heldData) - 3 + 1):
                        t = ''.join(heldData[x:x + 3])
                        if (t in trigram.keys()):
                            x = trigram[t]
                        else:
                            x = 0
                        if t[0:2] in bigram.keys():
                            prob2 = x / (bigram[t[0:2]])
                        else:
                            prob2 = 0
                        if (t[0:2] in bigram.keys()):
                            x = bigram[t[0:2]]
                        else:
                            x = 0
                        if t[0:1] in unigram.keys():
                            prob1 = x / unigram[t[0:1]]
                            prob = unigram[t[0:1]] / sum(unigram.values())
                        else:
                            prob1 = 0
                            prob = unigram["#"] / sum(unigram.values())
                        p = a * prob + b * prob1 + c * prob2
                        probability.append(p)
                    size=len(heldData) - 2
                    perplexity=getLeveragePerplexity(probability,size)
                    if(perplexity<perplex):
                        perplex=perplexity
                        lam1=a
                        lam2=b
                        lam3=c
    return lam1,lam2,lam3

################################################################################
# Get Perplexity for each test file
################################################################################
def getLeveragePerplexity(probability, l):
    perplexity = 0
    for p in probability:
      perplexity = math.log(p) + perplexity

    perplexity = (math.exp(-1 * (perplexity / l)))
    perplexity_leverage.append(perplexity)
    return perplexity

################################################################################################
# Preprocessing of training data
# removed blank lines, replace newline chars with space, replaced multiple spaces with one space
# removed punctuations
# Replaced characters with count<=5 with '#', as UNK will give rise to undesired U|N, N|K
################################################################################################
def processAllTrainingFiles(path):
    text = ""
    for data in os.listdir(path):
        file = open(path + data, 'r')
        text = text + file.read()
        #text = text.rstrip()
        text = text.replace('\n', ' ')
        text=text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(" +", " ", text)
    d = {}
    for i in range(len(text)):
        d.setdefault(text[i], 0)
        d[text[i]] += 1
    for num in d.keys():
        if d[num] <= 5:
            # Here I am using # instead of UNK to avoid U|N, N|K,
            # # is not present in training and test data so chose it
            text = text.replace(str(num), "#")
    return text


#############################################################################
#getting ngrams from 100% training data
#############################################################################
def findNGramCounts(text):
    trainUnigram = {}
    trainBigram = {}
    trainTrigram = {}
    for x in range(len(text) - 3 + 1):
        t = ''.join(text[x:x + 3])
        trainTrigram.setdefault(t, 0)
        trainTrigram[t] += 1

    for y in range(len(text) - 2 + 1):
        b = ''.join(text[y:y + 2])
        trainBigram.setdefault(b, 0)
        trainBigram[b] += 1

    for z in range(len(text)):
        u = ''.join(text[z:z + 1])
        trainUnigram.setdefault(u, 0)
        trainUnigram[u] += 1

    return trainUnigram,trainBigram,trainTrigram

####################################################################
#process entire test data
####################################################################
def processTestData():
   testPath =os.getcwd()+"\\test_data\\"
   test=""
   data=[]
   for l in os.listdir(testPath):
       file = open(testPath + l, 'r',encoding='UTF8')
       test=file.read()
       #test = test.rstrip()
       test=test.replace('\n',' ')
       test=test.translate(str.maketrans('', '', string.punctuation))
       test = re.sub(" +", " ", test)
       data.append(test)

   d = {}
   for i in range(len(data)):
       d=dict.fromkeys(data[i], 0)
       for c in data[i]:
           d[c] += 1
   for i in range(len(data)):
       for num in d.keys():
         if d[num] <= 5:
           # Here I am using # instead of UNK to avoid U|N, N|K
           data[i] = data[i].replace(str(num), "#")
   return data

########################################################################################
#get probabilities for test data
########################################################################################
def getProbabilityTestData(trainUnigram,trainBigram,trainTrigram,data,lam1,lam2,lam3):

    for i in range(len(data)):
        text=""
        probability = []
        text=data[i]
        p = Decimal(1.0)
        for x in range(len(text) - 3 + 1):
            t = ''.join(text[x:x + 3])
            if (t in trainTrigram.keys()):
                x = trainTrigram[t]
            else:
                x = 0
            if t[0:2] in trainBigram.keys():
                prob2=x/(trainBigram[t[0:2]])
            else:prob2=0
            if (t[0:2] in trainBigram.keys()):
                x = trainBigram[t[0:2]]
            else:x = 0
            if t[0:1] in trainUnigram.keys():
                prob1=x/trainUnigram[t[0:1]]
                prob = trainUnigram[t[0:1]] / sum(trainUnigram.values())
            else:
                prob1=0
                prob=trainUnigram["#"]/sum(trainUnigram.values())

            p=lam1*prob+lam2*prob1+lam3*prob2
            probability.append(p)
        getPerplexity(probability,len(text)-2)

################################################################################
#Get Perplexity for each test file
################################################################################
def getPerplexity(probability,l):
    perplexity = 0
    for p in probability:
        perplexity = math.log(p) + perplexity

    perplexity=(math.exp(-1 * (perplexity / l)))
    perplexity_trigram.append(perplexity)
    return perplexity
####################################################################################
#Writing Perplexities to file
####################################################################################
def writePerplexityToFile(perplexity_trigram):
    file_list=[]
    path = os.getcwd() + "\\test_data\\"
    for data in os.listdir(path):
        file_list.append(data)
    with open(os.getcwd() +"\\output3\\PerplexitiesInterpolation.txt", "w") as l:
       for j, test in zip(range(0,len(perplexity_trigram)), file_list):
               l.write(test+","+str(perplexity_trigram[j]))
               l.write("\n")
               dictPerpFileName[test]=perplexity_trigram[j]
    l.close()
####################################################################################
#Getting Top 50 Perplexities
####################################################################################
def topFiftyPerplexities(dictPerpFileName,perplexity_trigram):
    with open(os.getcwd()+"\\output3\\Top50_Interpolation.txt", "w") as l:

        for i,j in zip(range(50),(sorted(dictPerpFileName, key=dictPerpFileName.__getitem__,reverse=True))):
            if(i<=50):
                l.write(j+" "+str(dictPerpFileName[j]))
                l.write("\n")
    l.close()

if __name__ == "__main__": main()