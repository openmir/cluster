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

TOOLSDIR = os.getenv('TOOLSDIR', '/global/scratch/sness/openmir/tools')
TMPDIR = os.getenv('TMPDIR', '/global/scratch/sness/openmir/tmp')

nextractPath = os.path.join(TOOLSDIR, "marsyas/release/bin/nextract")
trainPath = os.path.join(TOOLSDIR, "liblinear-1.93/train")                             
predictPath = os.path.join(TOOLSDIR, "liblinear-1.93/predict")                             
libsvmScalePath = os.path.join(TOOLSDIR, "libsvm-3.17/svm-scale")


def parseInput(inFilename):
    data = []
    with open(inFilename) as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def generateFilenames(runs):
    outputDir = 'output/features'
    for run in runs:
        run['extractFilename'] = '%s'
        run['scaledFilename'] = '%s.scaled' % run['extractFilename']

    return runs

def run(runs):
    print "TOOLSDIR=%s" % (TOOLSDIR)
    print "TMPDIR=%s" % (TMPDIR)
    for run in runs:
        print run

    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: run-nextract-liblinear.py runs.txt bextract.mf"
        sys.exit(1)

    inFilename = sys.argv[1]
    runs = parseInput(inFilename)
    runs = generateFilenames(runs)
    run(runs)
        

