""" DOCSTRING """
from collections import Counter

#import os
import re
import io
import math as calc
#import sys
from data_processing import string2ngrams

class NaiveBayesClassifier():
    """A program to use machine learning techniques with ngram maximum likelihood probabilistic language models as feature to build a bayesian text classifier."""

    def __init__(self, labels, files, words, codec):
        """Constructor method to load training data, and train classifier."""
        #self.ng = nGram(False, False)
        self.codec = codec
        self.labels = labels
        self.words = words
        self.files = files
        
        # print(self.words)
        #print(sys.getdefaultencoding())

        self.train()
        return

    def train(self):
        """Method to train classifier by calculating Prior and Likelihood."""
        self.prior = self.calculatePrior()
        self.unigram = self.createUnigram()

    def classify(self, document, ng=0):
        """Method to load test data and classify using training data."""
        words = ""
        #print("opening: " + document)
        with io.open(document, 'r', encoding=self.codec) as t_f:
            for line in t_f:
                words = words + ' ' + line

        words = re.findall(r"<?/?\w+>?", words.lower())
        #print("###### Testing with ######")
        #print(string2ngrams("_".join(words),2))

        if ng == 2:
            words = string2ngrams("_".join(words),2)
        elif ng == 3:
            words = string2ngrams("_".join(words),3)

        P = dict.fromkeys(self.labels, 0)
        for label in self.labels:
            for word in words:
                P[label] = P[label] + self.calculateLikelihood(word, label)
        P[label] = P[label] + calc.log(self.prior[label])
        #print(sorted(P, key=P.get, reverse=True)[0])
        return sorted(P, key=P.get, reverse=True)[0]

    def calculatePrior(self):
        """Method to calculate Prior."""
        prior = dict()
        for label in self.labels:
            prior[label] = len(self.files[label])
        s = sum(prior.values())
        for label in prior.keys():
            prior[label] = prior[label]/s
        return prior

    def calculateLikelihood(self, word, label):
        """Method to calculate Likelihood."""
        return self.unigramProbability(word, label)

    def unigramProbability(self, word, label):
        """Method to calculate Unigram Maximum Likelihood Probability with Laplace Add-1 Smoothing."""
        return calc.log((self.unigram[label][word]+1)/(len(self.words[label])+len(self.unigram[label])))

    def createUnigram(self):
        """Method to create Unigram for each class/label."""
        unigram = dict.fromkeys(self.labels, dict())
        unigram_f = dict.fromkeys(self.labels, dict())
        for label in self.labels:
            unigram[label] = Counter(self.words[label])
            unigram_f[label] = Counter()
            for item in unigram[label]:
                #(unigram[label][item])
                if unigram[label][item] > 50:
                    unigram_f[label][item] = unigram[label][item]
            print("## " + label + " ##")
            print("All -> More than 50: " + str(len(unigram[label])) + " > " + str(len(unigram_f[label])))
            print(unigram[label].most_common(5))

        return unigram_f
