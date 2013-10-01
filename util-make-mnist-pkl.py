#!/usr/bin/python

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
#DEBUG = False
DEBUG = True

NUM_COLUMNS = 784
TRAIN_ROWS = 50000
VALID_ROWS = 10000
TEST_ROWS = 10000


def runOne(inFilename,numRows,numColumns):
    dataSet = (np.zeros((numRows, numColumns), np.float32),np.zeros((numRows,), np.int64))
    
    infile = open(inFilename, "r")
    line = infile.readline()

    i = 0
    while line and i < numRows:
        a = line.split()
        dataSet[1][i] = (a[0])
        j = 1
        while j < len(a) and j < numColumns:
            dataSet[0][i][j-1] = (a[j].split(":")[1])
            j += 1

        line = infile.readline()
        i += 1

    if DEBUG:
        pp.pprint("***dataSet")
        pp.pprint(dataSet)
        print dataSet[0].shape
        print dataSet[0].dtype
        print dataSet[1].shape
        print dataSet[1].dtype

    return dataSet

def run(inTrainFilename,inValidFilename,inTestFilename,outFilename):
    trainData = runOne(inTrainFilename, TRAIN_ROWS, NUM_COLUMNS)
    validData = runOne(inValidFilename, VALID_ROWS, NUM_COLUMNS)
    testData = runOne(inTestFilename, TEST_ROWS, NUM_COLUMNS)

    # f = gzip.open(outFilename, 'wb')
    f = open(outFilename, 'wb')
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
        

