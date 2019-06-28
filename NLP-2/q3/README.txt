This folder contains the following:
1. test folder containing test data for Question 3.5
2. train folder containing pos and neg folders for training.
3. Question3.1.py containing solution for 3.1 and 3.2
4. Question3.3.py containing solution for 3.3.
5. Question3.4.py containing solution for 3.4.
6. Question3.5.py containing solution for 3.5.
7. output folder containing all output files for 3.1 to 3.4 and labels.zip for 3.5.
8. labels.zip contains pos.txt and neg.txt

How to run?

Copy all contents of this folder inside Scripts folder of PycharmProject along with GoogleNews-vectors-negative300.bin.gz file(file is not included in folder because of its huge size. Download all dependencies with conda or pip. Run each python file individually to get results. 

Optimization Process
# Optimization Parameters: I have used 4 types of activation functions
# activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
# Activation function for the hidden layer.
#
# ‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
# ‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
# ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
# ‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
#  The solver/optimizer that I am using is lbfgs, shuffling is not required for this solver.
# I am using number of nodes in first hidden layer between 45 and 50 for optimization process.
# I have a total of 24 combinations of the parameters and every iteration will run 10 fold times.
# so there are 240 iterations in total. The highest average accuracy parameters
# for any out of 24 combinations
# are taken for retraining for 3.2.
