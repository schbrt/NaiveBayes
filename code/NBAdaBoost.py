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

        self.correct_weights = []
        self.incorrect_weights = []
        #keeps track of pairs of (classifier, classifier weight)
        self.ensemble = []

    def sample(self):
        sample = np.random.choice(self.train_data, self.train_len, replace = True, p = self.weights)
        return sample

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
        for idx in range(len(self.weights)):
            self.weights[idx] = self.weights[idx]/tot

    def reset_weights(self):
        for item in weights:
            item = 1/self.train_len

    def error(self):
        sum = 0
        for i in self.incorrect_indicies:
            sum += self.weights[i]

        return sum

    def test(self, test_data, pos, neg, cl, error_rate):
        if error_rate < .5:
            test_ret = []
            count = 0
            self.correct_indicies = []
            self.incorrect_indicies = []

            for line in test_data:
                val = cl.classify(line, pos, neg)

                if val[0] != val[1]:
                    self.incorrect_indicies.append(count)

                else:
                    self.correct_indicies.append(count)

                test_ret.append(val)
                count += 1

            error_rate = self.error()
            self.update(error_rate, cl)

        else:
            self.reset_weights()



args = sys.argv
train_file = '../' + args[1]
test_file = '../' + args[2]


train_data = NB.read(train_file)
ab = AdaBoost(train_data)
error_rate = 0
#ye olde iteration begins here
for i in xrange(20):
    #create sample data set
    sample = ab.sample()
    #build classifier
    pos, neg, cl = NB.train(sample)

    ab.test(sample, pos, neg, cl, error_rate)

print ab.ensemble


