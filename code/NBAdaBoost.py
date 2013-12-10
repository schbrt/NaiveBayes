from __future__ import division
import NB
import numpy as np
import sys
import math
import copy
import random

class AdaBoost:
    def __init__(self, train_data):
        self.train_data = train_data
        self.train_len = len(train_data)
        self.weights = np.ones(self.train_len)/self.train_len

        self.correct_weights = []
        self.incorrect_weights = []
        #keeps track of pairs of (classifier, classifier weight)
        self.ensemble = []

    def fast_sample(self):
        sample = np.random.choice(self.train_data, self.train_len, replace = True, p = self.weights)
        return sample

    def sample(self):
        ret = []
        while len(ret) < len(self.train_data):
            choice = self.choose()
            ret.append(choice)
        return ret

    """
    Numpy was being a piece of shit, so I wrote a work around.  If it can successfully use np.random.choice
    for a data set, it does in the interest of speed (especially important for a1a dataset).  If not, it uses my random
    picker, which is much slower but works, which has to be worth something. Maybe.
    """
    def sample_picker(self, train_file):
        if 'cancer' in train_file or 'poker' in train_file:
            return self.sample()
        else:
            return self.fast_sample()

    def choose(self):
        pos = copy.copy(self.train_data)
        total_weight = sum(self.weights)
        r = random.uniform(0, total_weight)
        upto = 0
        for i in xrange(self.train_len):
          if upto + self.weights[i] > r:
             return pos[i]
          upto += self.weights[i]

    def update(self, error_rate, classifier, pos, neg):
        alpha = .5 * math.log((1-error_rate)/float(error_rate))

        for idx in self.correct_weights:
            mult = math.exp(alpha)
            self.weights[idx] *= mult
        for idx in self.incorrect_weights:
            mult = 1/(math.exp(alpha))
            self.weights[idx] *= mult

        self.normalize_weights()
        #2 gets rid of factor of .5
        self.ensemble.append((classifier, pos, neg, 2*alpha))

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
            self.update(error_rate, cl, pos, neg)

        else:
            self.reset_weights()

def test_ensemble(test_file):
    test_data = NB.read(test_file)

    test_TP = 0
    test_TN = 0
    test_FP = 0
    test_FN = 0

    for line in test_data:
        tot = 0
        choice = None
        label = line[0]
        for item in ab.ensemble:
            classifier = item[0]
            pos = item[1]
            neg = item[2]
            weight = item[3]
            val = classifier.classify(line, pos, neg)
            guess = int(val[1])

            tot += (weight*guess)

        if tot < 0:
            choice = '-1'
        else:
            choice = '+1'


        if label == '+1' and choice == '+1':
            test_TP += 1
        elif label == '-1' and choice == '-1':
            test_TN +=1
        elif label == '-1' and choice == '+1':
            test_FP += 1
        elif label == '+1' and choice == '-1':
            test_FN += 1

    print test_TP, test_FN, test_FP, test_TN


args = sys.argv
train_file = '../' + args[1]
test_file = '../' + args[2]


train_data = NB.read(train_file)
ab = AdaBoost(train_data)
error_rate = 0
#ye olde iteration begins here
for i in xrange(20):
    #create sample data set
    sample = ab.sample_picker(train_file)
    #build classifier
    pos, neg, cl = NB.train(sample)

    ab.test(sample, pos, neg, cl, error_rate)



test_ensemble(train_file)
test_ensemble(test_file)

