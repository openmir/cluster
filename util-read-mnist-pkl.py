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

def run(dataset):
    
    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    pp.pprint("***train_set")
    pp.pprint(train_set)
    print train_set[0].shape

    pp.pprint("***valid_set")
    pp.pprint(valid_set)
    print valid_set[0].shape

    pp.pprint("***test_set")
    pp.pprint(test_set)
    print test_set[0].shape

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print "Usage: util-read-mnist-pkl.py file.pkl.gz"
        sys.exit(1)

    inFilename = sys.argv[1]        
    run(inFilename)
        

