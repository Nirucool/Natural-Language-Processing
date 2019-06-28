import os
import re
import string

###################################################################################
# main method/driver method#
# keep training data folder gutenberg in current working directory,
# remove readme file present in it.
####################################################################################
def main():
    path = os.getcwd() + "\\gutenberg\\"
    processTrainingFiles(path)

################################################################################################
# Preprocessing of training data-Q3.1
# removed blank lines, replace newline chars with space, replaced multiple spaces with one space
# removed punctuations
# Replaced characters with count<=5 with '#', as UNK will give rise to undesired U|N, N|K
################################################################################################
def processTrainingFiles(path):
    text=""
    for data in os.listdir(path):
                file = open(path + data, 'r')
                text=text+file.read()
                text=text.replace('\n',' ')
                text=text.translate(str.maketrans('', '', string.punctuation))
                text=re.sub(" +", " ", text)
    d={}
    for i in range(len(text)):
        d.setdefault(text[i],0)
        d[text[i]] += 1
    for num in d.keys():
        if d[num]<=5:
           # Here I am using '#' instead of UNK to avoid U|N, N|K etc
           # '#' is not present in training data and test data
            text=text.replace(str(num),"#")
    findNGramCounts(text)
###############################################################################################
# find unigram bigram and trigram counts and writing them to files
# create output3 folder in current working directory
# unigrams are written in unigramCount.txt in output3 folder
# bigrams are written in bigramCount.txt in output3 folder
# trigrams are written in trigramCount.txt in output3 folder
###############################################################################################
def findNGramCounts(text):
    trigram = {}
    bigram={}
    unigram={}

    for x in range(len(text) - 3 + 1):
        t = ''.join(text[x:x + 3])
        trigram.setdefault(t, 0)
        trigram[t] += 1

    for y in range(len(text) - 2 + 1):
        b = ''.join(text[y:y + 2])
        bigram.setdefault(b, 0)
        bigram[b] += 1

    for z in range(len(text)):
        u = ''.join(text[z:z + 1])
        unigram.setdefault(u, 0)
        unigram[u] += 1

    with open(os.getcwd() + "\\output3\\unigramCount.txt", "w") as u:
        u.write(str(unigram))
    u.close()
    with open(os.getcwd() + "\\output3\\bigramCount.txt", "w") as b:
        b.write(str(bigram))
    b.close()
    with open(os.getcwd() + "\\output3\\trigramCount.txt", "w") as t:
        t.write(str(trigram))
    t.close()


if __name__ == "__main__": main()