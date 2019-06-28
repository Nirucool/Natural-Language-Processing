This folder has following contents:
1. train folder containing Test_Data.csv and Train_Data.csv.
2. Question4.py containing solution for Question 4.1 and Question 4.2
3. Question4.3.py containing solution for Question 4.3 and Question 4.4
4. GoogleNews-vectors-negative300.bin.gz file is not included in the folder as its size is too large

How to run?

Copy the contents of the folder along with GoogleNews-vectors-negative300.bin.gz file and paste them in Scripts folder of PycharmProject.
Run each question individually.Download all dependencies with conda or pip.

Results: Question 4.1 and Question 4.2
With given features
0.22461729023458354
(0.673167066845756, 9.354069464157302e-28)
With pairwise kernel similarity
0.21585679358762774
(0.6944329487211355, 4.0559043109817306e-30)
With maximum repetition of trigrams 
0.21894478133177522
(0.6775029132194252, 3.2014277319324223e-28)

 # Question 4.2
    # feature 1
    # PAIRWISE KERNELS - Kernels are measures of similarity, i.e. s(a, b) > s(a, c) if objects
    # a and b are considered “more similar” than objects a and c. A kernel must also be positive
    # semi-definite.
    # Here, the pairwise kernel similarity is found using word2vec vectors between all sentences
    # in a summary. The max value is taken into account as feature for calculating MSE and Pearson Correlation
    # Coefficient. More the similarity, more is the redundancy. Hence it improves MSE and Pearson
    # correlation coefficient.

    # feature-2 Maximum repetition of trigrams
    # More is the repetition of trigrams, more will the repetition count which reflects
    # redundancy and less count reflects less redundancy or non redundancy. Hence it        # improves MSE and Pearson correlation coefficient values for redundancy.




Results: Question 4.3 and Question 4.4
With given features
# 0.22993706600411773
# (0.3521331207011803, 3.163607855250048e-07)
With SMOG Index
# 0.22948742171238876
# (0.3547109953705656, 2.5568576898239954e-07)
With LIX Formula
# 0.22856596250744246
# (0.3545475618365523, 2.591754092042154e-07)
#######################################################################################
    # Question 4.4
    # feature 1
    # Maximum value of SMOG index,a Simple Measure of Gobbledygook
    # The value of SMOG index ranges from 1 to 240 with the higher value for less readable or
    # less fluent. It uses words with more than 3 syllables to determine complexity of the sentence.
    # This gives a measure of the fluency or readability or understandability of the summaries
    # and hence reduces the MSE and increases Pearson Correlation Coefficient
#######################################################################################

######################################################################################
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
#######################################################################################



