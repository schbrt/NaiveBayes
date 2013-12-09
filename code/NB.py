from __future__ import division

class Class:
    def __init__(self, label, count, freqs, tot, feats):
        self.label = label
        self.count = count
        self.freqs = freqs
        self.probs = {}
        #used to calculate prior
        self.tot = tot
        self.prior = None
        self.feats = feats

    def set_prior(self):
        self.prior = self.count/self.tot

    def set_probs(self):
        num = self.count
        for freq in self.freqs:
            self.probs[freq] = ((self.freqs[freq] + 1)/(num + 1)) #* self.prior

class Classifier:
    def __init__(self, data):
        self.data = data
        self.neg_num = 0
        self.pos_num = 0
        #all features across set?
        self.features = set()

        self.pos_feats = {}
        self.neg_feats = {}

        self.test_pos_num = 0
        self.test_neg_num = 0

    def find_features(self):
        for item in self.data:
            clean = item[1:]
            for val in clean:
                feat = val.split(':')[0]
                self.features.add(feat)
    """
    Check if key exists in positive feature dictionary
    """
    def pos_key_exists(self, key):
        return key in self.pos_feats

    """
    Check if key exists in negative feature dictionary
    """
    def neg_key_exists(self, key):
        return key in self.neg_feats

    """
    Populates the dictionaries
    """
    def occur(self):
        for item in self.data:
            info = item[1:]
            label = item[0]

            if label == '+1':
                self.pos_num += 1
                for item in info:
                    self.update_dict(item, self.pos_feats, '+1')

            elif label ==  '-1':
                self.neg_num += 1
                for item in info:
                    self.update_dict(item, self.neg_feats, '-1')


    def update_dict(self, item, curr_dict, sign):
        if sign == '-1':
             if self.neg_key_exists(item):
                curr_dict[item] += 1
             else:
                curr_dict[item] = 1
        elif sign == '+1':
            if self.pos_key_exists(item):
                curr_dict[item] += 1
            else:
                curr_dict[item] = 1


    """
    Classify lines
    """
    def classify(self, item, c1, c2):
        info = item[1:]
        label = item[0]

        if label == '+1':
            self.test_pos_num += 1
        elif label == '-1':
            self.test_neg_num += 1

        prob1 = c1.prior
        prob2 = c2.prior

        for key in info:
            #if key in c1.probs and key in c2.probs:
            if key in self.pos_feats:
                prob1 *= c1.probs[key]
            if key in self.neg_feats:
                prob2 *= c2.probs[key]

        pnt = ''
        guess = None
        if prob1 < prob2:
            guess = '-1'
        elif prob2 < prob1:
            guess = '+1'

        return (label, guess)

"""
Reads in the data sets
"""
def read(dataset):
    data = []
    with open(dataset) as f:
        curr_data = f.readlines()
        for line in curr_data:
            line.rstrip()
            curr = line.split()
            if len(curr) == 0:
                continue

            for item in curr[1:]:
                clean = item.split(':')[1]
                if clean == '0':
                    curr.remove(item)
            data.append(curr)
    return data

def train(train_data):

    cl = Classifier(train_data)
    cl.find_features()
    cl.occur()

    tot = cl.neg_num + cl.pos_num
    pos = Class("pos", cl.pos_num, cl.pos_feats, tot, cl.features)
    neg = Class("neg", cl.neg_num, cl.neg_feats, tot, cl.features)

    pos.set_prior()
    neg.set_prior()
    pos.set_probs()
    neg.set_probs()

    return pos, neg, cl

def test(test_data, pos, neg, cl):

    test_ret = []
    for line in test_data:
        val = cl.classify(line, pos, neg)
        test_ret.append(val)

    test_TP = 0
    test_TN = 0
    test_FP = 0
    test_FN = 0
    for item in test_ret:
        if item[0] == '+1' and item[1] == '+1':
            test_TP += 1
        elif item[0] == '-1' and item[1] == '-1':
            test_TN +=1
        elif item[0] == '-1' and item[1] == '+1':
            test_FP += 1
        elif item[0] == '+1' and item[1] == '-1':
            test_FN += 1

    print test_TP, test_FN, test_FP, test_TN


