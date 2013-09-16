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

TOOLSDIR = os.getenv('TOOLSDIR', '/global/scratch/sness/openmir/tools')
TMPDIR = os.getenv('TMPDIR', '/scratch/')

nextractPath = os.path.join(TOOLSDIR, "marsyas/release/bin/nextract")
liblinearTrainPath = os.path.join(TOOLSDIR, "liblinear-1.93/train")                             
liblinearPredictPath = os.path.join(TOOLSDIR, "liblinear-1.93/predict")                             
libsvmTrainPath = os.path.join(TOOLSDIR, "libsvm-3.17/svm-train")                             
libsvmPredictPath = os.path.join(TOOLSDIR, "libsvm-3.17/svm-predict")                             
libsvmScalePath = os.path.join(TOOLSDIR, "libsvm-3.17/svm-scale")
wekaPath = "java -classpath %s" % (os.path.join(TOOLSDIR, "weka/weka.jar"))

#DEBUG = False
DEBUG = True

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
        run['randHash'] = "%032x" % random.getrandbits(128)
        # TODO(sness) - Add parameters to nextract and svm to baseFilename
        run['baseFilename'] = os.path.join(TMPDIR,'nextract-svm-%s' % (run['randHash']))
        run['extractFilename'] = '%s.features' % (run['baseFilename'])
        run['arffFilename'] = '%s.arff' % (run['baseFilename'])
        run['scaleFilename'] = '%s.scaled' % (run['baseFilename'])
        run['scaleParamsFilename'] = '%s.params' % (run['baseFilename'])
        run['modelFilename'] = '%s.model' % (run['baseFilename'])
        run['predictionFilename'] = '%s.prediction' % (run['baseFilename'])

    return runs

def runExtract(run,inCollection):
    """ Extract audio features. """
    run['extractCommand'] = "%s %s %s -w %s -o %s" % (
        nextractPath, run['extractOptions'], inCollection, run['arffFilename'], run['extractFilename'])
    startTime = time.time()    
    run['extractOutput'] = commands.getoutput(run['extractCommand'])
    run['extractTime'] = time.time() - startTime

    if DEBUG:
        print "extractCommand=%s" % (run['extractCommand'])
        print "extractOutput=%s" % (run['extractOutput'])
        print "extractTime=%s" % (run['extractTime'])

def runScale(run):
    """ Scale the data with libsvm/scale. """
    if DEBUG:
        print "runScale"
    if run['scale'] == 'false':
        if DEBUG:
            print "Not scaling data"
        run['scaleFilename'] = run['extractFilename']
        return

    run['scaleCommand'] = "%s -s %s %s > %s" % (libsvmScalePath, run['scaleParamsFilename'], run['extractFilename'], run['scaleFilename'])
    startTime = time.time()
    run['scaleOutput'] = commands.getoutput(run['scaleCommand'])
    run['scaleTime'] = time.time() - startTime

    if DEBUG:
        print "scaleCommand=%s" % (run['scaleCommand'])
        print "scaleOutput=%s" % (run['scaleOutput'])
        print "scaleTime=%s" % (run['scaleTime'])

def runTrain(run):
    """ Train a model with a classifier. """
    if DEBUG:
        print "runTrain"
        
    # TODO(sness) - Change to allow libsvm or liblinear to be used
    if run['svm'] == 'libsvm':
        trainPath = libsvmTrainPath
    elif run['svm'] == 'weka':
        run['trainTime'] = 0.0
        return
    else:
        trainPath = liblinearTrainPath

    run['trainCommand'] = "%s %s %s %s" % (trainPath, run['svmOptions'], run['scaleFilename'], run['modelFilename'])
    startTime = time.time()
    run['trainOutput'] = commands.getoutput(run['trainCommand'])
    run['trainTime'] = time.time() - startTime

    if DEBUG:
        print "trainCommand=%s" % (run['trainCommand'])
        print "trainOutput=%s" % (run['trainOutput'])
        print "trainTime=%s" % (run['trainTime'])


def runPredict(run):
    """ Predict input data with a trained model. """
    if DEBUG:
        print "runPredict"

    weka = False
    if run['svm'] == 'libsvm':
        predictPath = libsvmPredictPath
    elif run['svm'] == 'weka':
        predictPath = wekaPath
        weka = True
    else:
        predictPath = liblinearPredictPath

    if weka == True:
        run['predictCommand'] = "%s %s -t %s" % (predictPath, run['svmOptions'], run['arffFilename'])
    else:
        run['predictCommand'] = "%s %s %s %s" % (predictPath, run['scaleFilename'], run['modelFilename'], run['predictionFilename'])

    startTime = time.time()
    run['predictOutput'] = commands.getoutput(run['predictCommand'])
    run['predictTime'] = time.time() - startTime

    if weka == True:
        m = re.search('=== Error on test data ===\s+Correctly Classified Instances\s+[0-9]+\s+([0-9.]+)', run['predictOutput'])
        if m is not None:
            run['predictAccuracy'] = float(m.group(1))
        else:
            run['predictAccuracy'] = -1.
    else:
        m = re.search('Accuracy = ([0-9.]+)', run['predictOutput'])
        if m is not None:
            run['predictAccuracy'] = float(m.group(1))
        else:
            run['predictAccuracy'] = -1.
    
    if DEBUG:
        print "predictCommand=%s" % (run['predictCommand'])
        print "predictOutput=%s" % (run['predictOutput'])
        print "predictTime=%s" % (run['predictTime'])
    

def removeTmpFiles(run):
    if os.path.exists(run['extractFilename']):
        os.remove(run['extractFilename'])
        
    if os.path.exists(run['scaleFilename']):
        os.remove(run['scaleFilename'])
        
    if os.path.exists(run['modelFilename']):
        os.remove(run['modelFilename'])

    if os.path.exists(run['predictionFilename']):
        os.remove(run['predictionFilename'])
        
    
def run(runs,inCollection):
    if DEBUG:
        print "TOOLSDIR=%s" % (TOOLSDIR)
        print "TMPDIR=%s" % (TMPDIR)
        
    for run in runs:
        runExtract(run,inCollection)
        runScale(run)
        runTrain(run)
        runPredict(run)
        print "|%s|%s|%s|%.2f|%.2f|%.2f|%s|" % (run['table'], run['extractOptions'], run['svmOptions'], run['extractTime'], run['trainTime'], run['predictTime'], run['predictAccuracy'])
        #removeTmpFiles(run)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: run-nextract-svm.py input/runsvm.test.json input/train.mf"
        sys.exit(1)

    inCommandFilename = sys.argv[1]
    inCollection = sys.argv[2]
    runs = parseInput(inCommandFilename)
    runs = generateFilenames(runs)
    run(runs,inCollection)
        

