import os
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
import re
import dropbox
import math
import string

LAMBDA=0.1
perplexity_trigram=[]
dictPerpFileName={}

###################################################################################
# main method/driver method#-Q3.3
# keep training data folder gutenberg in current working directory,
# remove Readme file present in it.
####################################################################################
def main():
   path = os.getcwd() + "\\gutenberg\\"
   # process entire training data
   text=processAllTrainingFiles(path)
   # get unigram, bigram and trigram from training data
   trainUnigram, trainBigram, trainTrigram = findNGramCounts(text)
   # process entire test data
   data=processTestData()
   # get perplexities of all files
   getPerplexity(data,trainBigram,trainTrigram)
   # write all perplexities to PerplexitiesADD_Lambda.txt
   writePerplexityToFile(perplexity_trigram)
   # get french files with 50 highest perplexities
   topFiftyPerplexities(dictPerpFileName,perplexity_trigram)


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
        text = text.replace('\n', ' ')
        text=text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(" +", " ", text)
    d = {}
    for i in range(len(text)):
        d.setdefault(text[i], 0)
        d[text[i]] += 1
    for num in d.keys():
        if d[num] <= 5:
            # Here I am using '#' instead of UNK to avoid U|N, N|K,
            # '#' is not present in training and test data so chose it
            text = text.replace(str(num), "#")
    return text

#############################################################################
#getting ngrams from 100% training data
#############################################################################
def findNGramCounts(text):
    trainTrigram = {}
    trainBigram={}
    trainUnigram={}

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
           # Here I am using # instead of UNK to avoid U|N, N|K,
           # # is not present in training and test data so chose it
           data[i] = data[i].replace(str(num), "#")
   return data

####################################################################
#get perplexities for each test file using ADD lambda
####################################################################
def getPerplexity(data,trainBigram,trainTrigram):

       for i in range(len(data)):
            text = ""
            text=data[i]
            p = perplexity(trainBigram, trainTrigram,text)
            perplexity_trigram.append(p)
       return perplexity_trigram

#####################################################################
# get perplexity for a file
#####################################################################
def perplexity(trainBigram,trainTrigram,text):
    perplexity = 0
    for x in range(len(text) - 3 + 1):
        t = ''.join(text[x:x + 3])
        if (t[0:2] in trainBigram.keys()):
           y=trainBigram[t[0:2]]
        else: y=0
        if(t not in trainTrigram.keys()):
            x=0
        else:x=trainTrigram[t]
        prob=(x+0.1)/(y+(0.1*len(trainTrigram)))
        perplexity = math.log(prob) + perplexity

    perplexity=math.exp(-1 * (perplexity / (len(text)-2)))
    print(perplexity)
    return perplexity

####################################################################################
#Writing Perplexities to file
####################################################################################
def writePerplexityToFile(perplexity_trigram):
    file_list = []
    path = os.getcwd() + "\\test_data\\"
    for data in os.listdir(path):
        file_list.append(data)
    with open(os.getcwd() + "\\output3\\PerplexitiesADD_Lambda.txt", "w") as l:
        for j, test in zip(range(0, len(perplexity_trigram)), file_list):
            l.write(test + "," + str(perplexity_trigram[j]))
            l.write("\n")
            dictPerpFileName[test] = perplexity_trigram[j]
    l.close()

####################################################################################
#Getting Top 50 Perplexities
####################################################################################
def topFiftyPerplexities(dictPerpFileName,perplexity_trigram):
    with open(os.getcwd()+"\\output3\\Top50_ADD_Lambda.txt", "w") as l:

        for i,j in zip(range(50),(sorted(dictPerpFileName, key=dictPerpFileName.__getitem__,reverse=True))):
            if(i<=50):
                l.write(j+" "+str(dictPerpFileName[j]))
                l.write("\n")
    l.close()

if __name__ == "__main__": main()