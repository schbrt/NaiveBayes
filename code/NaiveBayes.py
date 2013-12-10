import sys
import NB

#python NaiveBayes.py /Users/michael/Classwork/cs412/schuber7_assign4/a1a.train /Users/michael/Classwork/cs412/schuber7_assign4/a1a.test
args = sys.argv

train_file = '../' + args[1]
test_file = '../' + args[2]

train_data = NB.read(train_file)
test_data = NB.read(test_file)

pos, neg, cl = NB.train(train_data)
NB.test(train_data, pos, neg, cl)
NB.test(test_data, pos, neg, cl)

