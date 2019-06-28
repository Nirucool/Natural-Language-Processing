import os
import re
import numpy as np

wordTag = []
wordTagFreq = {}
tagList = []
wordList = []
tagBigram = {}
tagUnigram = {}
emissionProb = {}
newWord = {}


###################################################################################
# main method/driver method#
# keep training data folder brown in current working directory,
# remove cats.txt file present in it.
# threshold for UNK words is kept as 1. word will be marked as UNK if it is present
# just once
####################################################################################
def main():
    path = os.getcwd() + "\\brown\\"
    ####Q 4.1################################
    for file in os.listdir(path):
        data = open(path + file, 'r')
        #adding start and end tags
        line = "<s>/start " + (data.read())
        while ("\n\n") in line:
            line = line.replace('\n\n', '\n')
        line = line.replace('\n', ' <**s>/end  <s>/start ')
        line = line.replace("\t", " ")
        while ("  ") in line:
            line=re.sub("  ", " ", line)
        tags=line.split(" ")
        for z in tags:
            wordTag.append(z)
    ##################################################
    #getting list of tags
    for k in wordTag:
        try:
            s = k.split("/")[1]
        except:
            continue;
        tagList.append(s)
    ##################################################
    #getting list of words
    for j in wordTag:
        try:
            k = j.split("/")[0]
        except:
            continue;
        wordList.append(k)
    ####################################################
    #getting list of words with count above threshold
    word = {}
    for z in wordList:
        word.setdefault(z, 0)
        word[z] += 1
    for c in word.keys():
        if word[c] > 1:
            newWord.setdefault(c, 0)
            newWord[c] = word[c]
    #####################################################
    #getting list of word-tag pairs after marking words below
    # and equal to threshold count as UNK
    newWordTag = []
    for i in range(len(wordTag)):
        try:
            if wordTag[i].split('/')[0] in newWord.keys():
                newWordTag.append(wordTag[i])
            else:
                x = "UNK/" + wordTag[i].split('/')[1]
                newWordTag.append(x)
        except:
            continue
    ################################################################################
    #getting frequency count of word-tag pairs and write it to wordTagCount.txt file
    for i in range(len(newWordTag)):
        wordTagFreq.setdefault(newWordTag[i], 0)
        wordTagFreq[newWordTag[i]] += 1
    f = open(os.getcwd() + "\\output\\wordTagCount.txt", "w")
    for k in wordTagFreq.keys():
        f.write(k + "," + str(wordTagFreq[k]))
        f.write("\n")
    f.close()
    tagUnigram = tagUnigramCount()
    tagBigram = tagBigramCount()


##############################################################
#tagUnigram Count C (ti)
##############################################################
def tagUnigramCount():
    for k in tagList:
        tagUnigram.setdefault(k, 0)
        tagUnigram[k] += 1
    f1 = open(os.getcwd() + "\\output\\tagUnigramCount.txt", "w")
    for l in tagUnigram.keys():
        f1.write(l + "," + str(tagUnigram[l]))
        f1.write("\n")
    f1.close()
    return tagUnigram

##############################################################
#tagBigram Count C (ti-1 , ti)
##############################################################
def tagBigramCount():
    for j in range(len(tagList) - 1):
        k = " ".join(tagList[j:j + 2])
        tagBigram.setdefault(k, 0)
        tagBigram[k] += 1
    f = open(os.getcwd() + "\\output\\tagBigramCount.txt", "w")
    for l in tagBigram.keys():
        f.write(l + "," + str(tagBigram[l]))
        f.write("\n")
    f.close()
    emissionProbability()
    transitionProbability()
    return tagBigram


##################################################################
#Transition Probability Q 4.2
##################################################################
def transitionProbability():
    transProb={}
    for j in tagBigram.keys():
        t1,t2=j.split(" ")
        try:
            probability=tagBigram[j]/tagUnigram[t1]
        except:continue
        transProb.setdefault(t2+"|"+t1,0)
        transProb[t2+"|"+t1]=probability
    file = open(os.getcwd() + "\\output\\transitionProbability.txt", "w")
    for l in transProb.keys():
        file.write(l + "," + str(transProb[l]))
        file.write("\n")

    file.close()
    randomSentences(transProb)
    posTagging(transProb)


#################################################################
#Emission Probability-Q 4.3
#################################################################
def emissionProbability():
    for j in wordTagFreq:
        try:
            eProbability=wordTagFreq[j]/tagUnigram[j.split("/")[1]]
        except:continue
        emissionProb.setdefault(j, 0)
        emissionProb[j]=eProbability
    file = open(os.getcwd() + "\\output\\emissionProbability.txt", "w")
    for l in emissionProb.keys():
        file.write(l + "," + str(emissionProb[l]))
        file.write("\n")
    file.close()

##################################################################
#POS tagging of Text_File.txt-Q 4.5
##################################################################
def posTagging(transProb):
    f = open(os.getcwd() + "\\Test_File.txt", "r")
    f1 = open(os.getcwd() + "\\output\\processedFile.txt", "w")
    prev=""
    for line in f.readlines():
        if (line.__contains__("< sentence ID =")):
            line = line.replace(line, "<s>\n")
        if (line.__contains__("<EOS>")):
            line = line.replace(line, "<**s>\n")
        f1.write(line)
        if (prev.__eq__("!\n") or prev.__eq__("?\n") or prev.__eq__(";\n") and prev.__eq__(line)):
                prev = ""
                f1.write("<**s1>\n")
                f1.write("<s1>\n")
        prev = line
    f.close()
    f1.close()
    f = open(os.getcwd() + "\\output\\processedFile.txt", "r")
    f1 = open(os.getcwd() + "\\output\\POSTagging.txt", "w")
    prevTag = "start"
    keyPrev= {"0_start": 1}
    state = ""
    i=0
    k=0
    for j, sentence in enumerate(f.readlines()):
        value=0
        temp=0
        sentence = sentence.replace("\n", "")
        if(sentence.__eq__("<s>") or sentence.__eq__("<s1>")):
            if (sentence.__contains__("<s>")):
                f1.write("< sentence ID =" + str(i + 1) + ">\n")
                i = i + 1
            prevTag = "start"
            keyPrev.update({str(j+1)+"_start":1})
            state="start"
            continue
        sen=""
        for l in tagList:
            if(sentence.__eq__("<**s1>")):
                sen="<**s>"
                e=sen + "/" + l
            else: e=sentence + "/" + l
            p=str(j) + "_" + prevTag
            t=l + "|" + prevTag
            if (e not in emissionProb):
                emissionProb.update({e: 0.00000000001})
            if (p in keyPrev and t in transProb and e in emissionProb):
                value=keyPrev[p] * transProb[t] * emissionProb[e]
            if (value > temp):
                state=l
                temp=value

        prevTag= state
        print(temp)
        print(" "+state)
        keyPrev.update({str(j+1) + "_" + state: temp})
        if(sentence.__contains__("<**s>")):
            f1.write("<EOS>\n")
        elif (not sentence.__eq__("<s>") and not sentence.__contains__("<**s>") and not sentence.__contains__("<**s1>")):
            f1.write(sentence + ", " + state + "\n")
    f1.close()
    f.close()


################################################################
# Random Sentence-Q-4.4
################################################################
def randomSentences(transProb):
        file = open(os.getcwd() + "\\output\\sentences.txt", "w")
        for count in range(0,5):
            # we need to output sentence, its tag and probability in a file
            sentence=[]
            senTags=[]
            sentProb = float(1.0)
            #start with tags whose previous tag was start and loop till end tag
            lastTag = "end"
            prevTag = "start"
            while prevTag is not lastTag:
                tags = []
                probTags = []
                keyWords = []
                wordProb = []
                #find out tags which has given prevTag
                for t in transProb.keys():
                    if t.split("|")[1]==prevTag and transProb[t] > 0:
                        probTags.append(transProb[t])
                        tags.append(t.split("|")[0])
                # we need to make the sum of probabilities one for random tag selection,
                # so put remaining probability for unknown word
                if (sum(probTags)<1):
                    unkProb=1-sum(probTags)
                    if "unknown" in tags:
                        probTags[tags.index("unknown")]=unkProb+probTags[tags.index("unknown")]
                    else:
                        tags.append("unknown")
                        probTags.append(unkProb)


                newState=np.random.choice(tags,1,p=probTags)
                prevTag = newState
                probNewState= probTags[tags.index(newState[0])]
                #stop the sentence if you reach end tag
                if (prevTag==lastTag):
                    break
                #get words on the basis of tags
                for e in emissionProb.keys():
                    try:
                        if (e.split("/")[1]==newState):
                            keyWords.append(e.split("/")[0])
                            wordProb.append(emissionProb[e])
                    except:continue

                newWord=np.random.choice(keyWords,1,p=wordProb)
                probNewWord=wordProb[keyWords.index(newWord[0])]
                #get sentence tags
                senTags.extend(prevTag)
                #get its probability
                sentProb=sentProb * probNewState * probNewWord
                #get sentence
                sentence.extend(newWord)

            #write to file
            file.write((" ").join(sentence))
            file.write("\n")
            file.write((" ").join(senTags))
            file.write("\n")
            file.write("Sentence"+str(count+1)+" Probability= " + str(sentProb))
            file.write("\n\n")
        file.close()

if __name__ == "__main__": main()
