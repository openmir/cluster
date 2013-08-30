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

toolsPath = "/global/scratch/sness/openmir/tools"
nextractPath = os.path.join(toolsPath, "marsyas/release/bin/nextract")
trainPath = os.path.join(toolsPath, "liblinear-1.93/train")                             
predictPath = os.path.join(toolsPath, "liblinear-1.93/predict")                             
libsvmScalePath = os.path.join(toolsPath, "libsvm-3.17/svm-scale")

def parseInput(inFilename):
    with open(inFilename) as f:
        data = json.load(f)
    return data

def generateFilenames(runs):
    outputDir = 'output/features'
    for run in runs:
        run['extractFilename'] = '%s/extract%s.libsvm' % (outputDir, run['extract'].strip().replace(" ",""))
        run['scaledFilename'] = '%s.scaled' % run['extractFilename']

    return runs

def runExtract(run):
    print "extract"

def runTrain(run):
    print "train"

def runPredict(run):
    print "predict"

def run(runs):
    # for run in runs:
    #     runExtract(run)
    #     runTrain(run)
    #     runPredict(run)

    print runs

            
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: run-nextract-liblinear.py runs.txt bextract.mf"
        sys.exit(1)

    inFilename = sys.argv[1]
    runs = parseInput(inFilename)
    runs = generateFilenames(runs)
    run(runs)
        

