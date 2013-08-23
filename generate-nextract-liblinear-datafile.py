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

toolsPath = "/global/scratch/sness/openmir/tools"
nextractPath = os.path.join(toolsPath, "marsyas/release/bin/nextract")
trainPath = os.path.join(toolsPath, "liblinear-1.93/train")                             
predictPath = os.path.join(toolsPath, "liblinear-1.93/predict")                             
libsvmScalePath = os.path.join(toolsPath, "libsvm-3.17/svm-scale")


def run(inFilename, runs):
    for run in runs:

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: generate-nextract-liblinear-datafile.py"
        sys.exit(1)

    runs = parseInput(inFilename)
    run(inFilename,runs)
        

