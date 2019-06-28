from nltk import word_tokenize
from nltk import tokenize,sent_tokenize
import string
import os
import re
from collections import Counter


####################################################################################
# in Stanford tokenizer a sentence ends when a sentence-ending character (., !, or ?)
# For dependency parse
# java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>dependencyParse.txt
# For CFG parse
# java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>cfgParse.txt
# For pos tagging
# java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "wordsAndTags" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>posTagged.txt
####################################################################################


def main():
    ################################################################################
    # Preprocessing all 100 files data-into file data.txt
    ################################################################################
    path = os.getcwd() + "\\corpus\\"
    data2=[]
    text=""
    files=[]
    for data in os.listdir(path):
        files.append(data)
        f=open(path+data,'r',encoding="UTF-8")
        text=text+f.read()
        f.close()
    text1=text

    lines=text1.split("\n")
    file7 = open(os.getcwd() + "\\output1\\data.txt", "w", encoding="UTF-8")
    x=""
    for line in lines:

        if(line.__contains__("?") or line.__contains__(".") or line.__contains__("!")):
           if(len(x)==0): x=x+line
           else: x=x+" "+line
           d=x
           b=x.translate(str.maketrans('', '', string.punctuation))
           x=""
        else:
            if (len(x) == 0):
                x = x + line
            else:
                x = x + " " + line
            continue
        datas=word_tokenize(b)
        if(len(datas)<=50):
           data2.append(d)
           file7.write(str(d)+" ")
    file7.close()

    ###############################################################################################
    #Question 1.2
    # Reported Count of sentences
    # 14004
    #
    # For CFG parse from command line:
    # java -Xmx12g -cp "*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz data.txt >>cfgParse.txt
    ###############################################################################################
    sentenceCount=0
    file1 = open(os.getcwd() + "\\output1\\cfgParse.txt", "r")
    sentenceCount+=file1.read().count('ROOT')
    print("sentence count:  "+ str(sentenceCount))
    file1.close()
    ###############################################################################################
    ###############################################################################################
    # Question 1.1
    #VB Verb, base form
    #VBD Verb, past tense
    #VBG Verb, gerund or present participle
    #VBN Verb, past participle
    #VBP Verb, non­3rd person singular present
    #VBZ Verb, 3rd person singular present

    #################################################################################
    countVerb = 0
    file = open(os.getcwd() + "\\output1\\posTagged.txt", "r",encoding="utf-8")
    for k in file.readlines():
        countVerb+=k.count("/VB")
    print("verb count:  "+ str(countVerb))
    file.close()
    #################################
    #average verb count per sentence
    #3.654812910596972
    #################################
    v=float(0.0);
    v=countVerb/sentenceCount
    print("average number of verbs per sentence:  "+ str(v))


########################################################################################
# Preprocessing 100 corpus files individually in //corpus-process folder for cfg and
# dependency parsing of individual files
########################################################################################
    for data, i in zip(os.listdir(path), range(100)):
        text = ""

        f = open(path + data, 'r', encoding="UTF-8")
        text = text + f.read()

        lines = text.split("\n")
        file7 = open(os.getcwd() + "\\corpus-process\\" + str(i)+"_"+ str(data), "w", encoding="UTF-8")
        x = ""
        for line in lines:
            if (line.__contains__("?") or line.__contains__(".") or line.__contains__("!")):
                if (len(x) == 0):
                    x = x + line
                else:
                    x = x + " " + line
                d = x
                b = x.translate(str.maketrans('', '', string.punctuation))
                x = ""
            else:
                if (len(x) == 0):
                    x = x + line
                else:
                    x = x + " " + line
                continue
            datas = word_tokenize(b)
            if (len(datas) <= 50):
                file7.write(str(d) + " ")
        file7.close()
    ####################################################################################################
    # Total preposition count per file using dependency parse
    # All individual 100 files after dependency parse are present in //corpus-out folder
    ####################################################################################################
    path2 = os.getcwd() + "\\corpus-out\\"
    prepCount=[]
    for data in os.listdir(path2):
        count=0
        f = open(path2 + data, 'r', encoding="UTF-8")
        count+=f.read().count("case(")
        prepCount.append(count)
        f.close()

    corp = open(os.getcwd() + "\\output1\\prepCountPerFile.txt", "w", encoding="UTF-8")
    for i in range(len(prepCount)):
        corp.write(str(files[i])+",  "+str(prepCount[i]))
        corp.write("\n")

    corp.close()
############################################################################################
# Question 1.3
#of , 10141
#in , 6250
#to , 3191
############################################################################################

    commonPrep = {}
    com = open(os.getcwd() + "\\output1\\dependencyParse.txt", "r")
    for line in com:
        line=str(line)
        a = word_tokenize(line)
        c=""
        if(len(a)>0):
          c=a[0]
        if(c.__contains__("case")):
            s=a[4].find("-")
            str1=a[4][0:s]
            commonPrep.setdefault(str1, 0)
            commonPrep[str1] += 1

    com.close()
    mostCommon = (Counter.most_common(commonPrep, 3))
    file3 = open(os.getcwd() + "\\output1\\commonPreposition.txt", "w")
    for pre in mostCommon:
        file3.write(str(pre[0]) + " , " + str(pre[1]))
        file3.write("\n")
    file3.close()





if __name__ == "__main__": main()