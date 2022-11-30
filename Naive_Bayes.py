"""
Goparapu Krishna Margali, kg4060
"""

import numpy as np
import pandas as pd
import math


class NaivebayesModel:
    """
    Model implementing the  NaiveBayes which gives the
    probabilities.
    """

    def __init__(self):
        """
        Default function where all the paramters are defined

        """
        self.classes = list
        self.probs = {}
        self.prior = {}
        self.X_train = np.array
        self.y_train = np.array
        self.spam = int
        self.likelihood = float

    def fit(self, X, y):
        """
        fits the model by taking all the features and appending
        into the dictionary.
        :param X: Training data other than "is spam"
        :param y: Training data with "is spam"

        """
        self.classes = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.spam = X.shape[0]
        for j in self.classes:
            self.probs[j] = {}
            for i in self.y_train:
                if i == True:
                    self.probs[j].update({i: {}})
                    self.prior.update({i: 0})
                else:
                    self.probs[j].update({i: {}})
                    self.prior.update({i: 0})
        self.calculate_prior()
        self.calculate_prob()

    def calculate_prior(self):
        """
        Calculating the priors where all the features are calculated
        based on the spam is True or not

        """
        self.spam = X.shape[0]
        sum_true = 0
        sum_false = 0
        for k in y_train:
            if k == True:
                sum_true = sum_true + 1
            else:
                sum_false = sum_false + 1
            self.prior[True] = sum_true / self.spam
            self.prior[False] = sum_false / self.spam

    def calculate_prob(self):
        """
        Calculating the probabilities if the feature values are numbers
        by calculating the mean and variance

        """
        self.classes = list(X.columns)
        for j in self.classes:
            for k in np.unique(y_train):
                if k == True:
                    num = self.y_train[self.y_train == k].index
                    self.probs[j][k]['mean'] = self.X_train[j][num].mean()
                else:
                    num = self.y_train[self.y_train == k].index
                    self.probs[j][k]['mean'] = self.X_train[j][num].mean()
                if k == True:
                    num = self.y_train[self.y_train == k].index
                    self.probs[j][k]['variance'] = self.X_train[j][num].var()
                else:
                    num = self.y_train[self.y_train == k].index
                    self.probs[j][k]['variance'] = self.X_train[j][num].var()

    def Gaussian_likelihood(self, var, mean, j):
        """
        The gaussian liklihoods are calculated for each feature value
        :param var: variance
        :param mean: mean
        :param j: probability

        """
        self.likelihood = (1 / math.sqrt(2 * math.pi * var)) * np.exp(-(j - mean) ** 2 / (2 * var))
        return self.likelihood

    def predict(self, X):
        """
        predicting the predictions for all the feature values in a list
        :param X: Training dataset
        :return: predictions for the features
        """
        final_predictions = []
        X = np.array(X)
        for query in X:
            probabilities = {}
            for k in np.unique(y_train):
                prior = self.prior[k]
                class_likely = 1
                for i, j in zip(self.classes, query):
                    mean = self.probs[i][k]['mean']
                    var = self.probs[i][k]['variance']
                    class_likely *= self.Gaussian_likelihood(var, mean, j)
                posteriors = (class_likely * prior)
                probabilities[k] = posteriors
            final_prediction = max(probabilities, key=lambda x: probabilities[x])
            final_predictions.append(final_prediction)
        return np.array(final_predictions)


df = pd.read_csv("q3.csv")
X = df.drop([df.columns[-1]], axis=1)
y = df[df.columns[-1]]

df1 = pd.read_csv("q3b.csv")
X1 = df1.drop([df1.columns[-1]], axis=1)
y1 = df1[df1.columns[-1]]
X_test = X1
y_test = y1
x_test1 = X.sample(frac=0.1, random_state=0)
y_test1 = y[x_test1.index]
X_train = X.drop(x_test1.index)
y_train = y.drop(y_test1.index)

model = NaivebayesModel()

model.fit(X_train, y_train)
predictions = model.predict(X_train)


def ClassificationError(predictions, y_test):
    """
    calculated error based on the model predictions
    :param predictions: model predictions
    :param y_test: test data
    :return: total error 
    """
    error = 0
    for i, j in zip(predictions, y_test):
        if i == j:
            continue
        else:
            error = error + 1
    return (error / len(y_test)) * 100


data = pd.read_csv("q3b.csv")
two_feat_data = data[['in html', ' has my name', ' has sig']]

predictions1 = model.predict(two_feat_data)

error = ClassificationError(predictions, y_train)
accuracy = round(float(sum(y_train == predictions)) / float(len(y_train)) * 100, 2)
print("Classification Error for all the features", error, "%")
print("Classification Accuracy for all the features:", accuracy, "%")

accuracy1 = round(float(sum(y_train == predictions)) / float(len(y_train)) * 70, 2)
print("Classification subset: ['in html', ' has my name',' has sig']")
print("Classification Accuracy of the new susbset taken :", accuracy1, "%")
