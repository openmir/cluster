#!/global/scratch/sness/openmir/tools/python/bin/python

#
#
#

import sys
import os
import datetime
import commands
import re
import time
import simplejson as json
import random
import numpy as np
import gzip
import cPickle

import pprint
pp = pprint.PrettyPrinter(indent=4)

MNIST_ZEROS = False
DEBUG = False

def runOne(inFilename):
    infile = open(inFilename, "r")
    line = infile.readline()
    data = []
    labels = []
    i = 0
    while line:
        a = line.split()
        labels.append(a[0])
        item = []
        for i in range(1,len(a)-1):
            item.append(a[i].split(":")[1])

        data.append(item)
        line = infile.readline()
        i += 1

    if MNIST_ZEROS:
        dataSet = (np.zeros((50000, 784), np.float32),np.zeros((50000,), np.int64))
    else:
        dataSet = (np.array(data, np.float32),np.array(labels, np.int64))

    if DEBUG:
        pp.pprint("***dataSet")
        pp.pprint(dataSet)
        print dataSet[0].shape
        print dataSet[0].dtype
        print dataSet[1].shape
        print dataSet[1].dtype
        
    return dataSet

def run(inTrainFilename,inValidFilename,inTestFilename,outFilename):
    trainData = runOne(inTrainFilename)
    validData = runOne(inValidFilename)
    testData = runOne(inTestFilename)

    f = gzip.open(outFilename, 'wb')
    cPickle.dump((trainData,validData,testData), f)
    f.close()
    

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "Usage: util-make-mnist-pkl.py inTrain.libsvm inValid.libsvm inTest.libsvm file.pkl.gz"
        print "Normal MNIST files have 50000 train, 10000 valid, 10000 test and have 784 feature vectors."
        sys.exit(1)

    inTrainFilename = sys.argv[1]        
    inValidFilename = sys.argv[2]        
    inTestFilename = sys.argv[3]        
    outFilename = sys.argv[4]        
    run(inTrainFilename,inValidFilename,inTestFilename,outFilename)
        

