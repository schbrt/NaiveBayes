from __future__ import division
import NB
import numpy as np
import sys
import math

class AdaBoost:
    def __init__(self, train_data):
        self.train_data = train_data
        self.train_len = len(train_data)
        self.weights = np.ones(self.train_len)/self.train_len
        self.rules = []

        self.correct_weights = []
        self.incorrect_weights = []
        #keeps track of pairs of (classifier, classifier weight)
        self.ensemble = []

    def sample(self):
        sample = np.random.choice(self.train_data, self.train_len, replace = True, p = self.weights)
        return sample

    def error(self):
        error = 0.0
        for i in xrange(train_len):

    def update(self, error_rate, classifier):
        alpha = .5 * math.log((1-error_rate)/float(error_rate))

        for idx in self.correct_weights:
            mult = math.exp(alpha)
            self.weights[idx] *= mult

        for idx in self.incorrect_weights:
            mult = 1/(math.exp(alpha))
            self.weights[idx] *= mult

        self.normalize_weights()
        #2 gets rid of factor of .5
        self.ensemble.append((classifier, 2*alpha))

    def normalize_weights(self):
        tot = float(sum(self.weights))
        for idx in range(len(weights)):
            weights[idx] = weights[idx]/tot

    def reset_weights(self):
        for item in weights:
            item = 1/self.train_len

    def test(test_data, pos, neg, cl, error_rate):
        if error rate < .5:
            test_ret = []
            count = 0
            self.correct_weights = []
            self.incorrect_weights = []

            for line in test_data:
                val = cl.classify(line, pos, neg)

                if val[0] != val[1]:
                    self.incorrect_weights.append(count)
                else:
                    self.correct_weights.append(count)

                test_ret.append(val)
                count += 1

            self.update(error_rate ,cl)

        else:
            self.reset_weights

args = sys.argv
train_file = '../' + args[1]
test_file = '../' + args[2]


train_data = NB.read(train_file)
AB = AdaBoost(train_data)

#ye olde iteration begins here
#create sample data set
sample = AB.sample()
#build classifier
pos, neg, cl = NB.train(sample)

ab.test(sample, pos, neg, cl, error_rate)


