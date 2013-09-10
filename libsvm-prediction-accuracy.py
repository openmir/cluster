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

def run(inFeaturesFilename,inPredictionFilename):
    inFeaturesFile = open(inFeaturesFilename, "r")
    inPredictionFile = open(inPredictionFilename, "r")
    
    featuresLine = inFeaturesFile.readline()
    predictionLine = inPredictionFile.readline()

    total = 0
    matches = 0
    while featuresLine:
        featuresLabel = int(featuresLine.split(" ")[0])
        predictionLabel = int(predictionLine.split(" ")[0])

        if (featuresLabel == predictionLabel):
            matches += 1 
        total += 1

        featuresLine = inFeaturesFile.readline()
        predictionLine = inPredictionFile.readline()

    accuracy = (float(matches) / float(total)) * 100.0
    print "Accuracy = %f%% (%i/%i)" % (accuracy,matches,total)

        

    
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: libsvm-prediction-accuracy.py inFeaturesFilename.txt inPredictionFilename.txt"
        sys.exit(1)

    inFeaturesFilename = sys.argv[1]
    inPredictionFilename = sys.argv[2]
    run(inFeaturesFilename,inPredictionFilename)
        

