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

DEBUG = False
#Debug = True

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
        run['extractTrainFilename'] = '%s.train.features' % (run['baseFilename'])
        run['extractTestFilename'] = '%s.test.features' % (run['baseFilename'])
        run['arffTrainFilename'] = '%s.train.arff' % (run['baseFilename'])
        run['arffTestFilename'] = '%s.test.arff' % (run['baseFilename'])
        run['scaleTrainFilename'] = '%s.train.scaled' % (run['baseFilename'])
        run['scaleTestFilename'] = '%s.test.scaled' % (run['baseFilename'])
        run['scaleParamsFilename'] = '%s.params.scaled' % (run['baseFilename'])
        run['modelFilename'] = '%s.model' % (run['baseFilename'])
        run['predictionFilename'] = '%s.prediction' % (run['baseFilename'])

    return runs

def runExtract(run,inTrainCollection,inTestCollection):
    """ Extract audio features. """
    run['extractTrainCommand'] = "%s %s %s -w %s -o %s" % (
        nextractPath, run['extractOptions'], inTrainCollection, run['arffTrainFilename'], run['extractTrainFilename'])
    startTime = time.time()    
    run['extractTrainOutput'] = commands.getoutput(run['extractTrainCommand'])
    run['extractTrainTime'] = time.time() - startTime

    run['extractTestCommand'] = "%s %s %s -w %s -o %s" % (
        nextractPath, run['extractOptions'], inTestCollection, run['arffTestFilename'], run['extractTestFilename'])
    startTime = time.time()    
    run['extractTestOutput'] = commands.getoutput(run['extractTestCommand'])
    run['extractTestTime'] = time.time() - startTime
    
    if DEBUG:
        print "extractTrainCommand=%s" % (run['extractTrainCommand'])
        print "extractTrainOutput=%s" % (run['extractTrainOutput'])
        print "extractTrainTime=%s" % (run['extractTrainTime'])
        print "extractTestCommand=%s" % (run['extractTestCommand'])
        print "extractTestOutput=%s" % (run['extractTestOutput'])
        print "extractTestTime=%s" % (run['extractTestTime'])

def runScale(run):
    """ Scale the data with libsvm/scale. """
    if DEBUG:
        print "runScale"
    if run['scale'] == 'false':
        if DEBUG:
            print "Not scaling data"
        run['scaleTrainFilename'] = run['extractTrainFilename']
        run['scaleTestFilename'] = run['extractTestFilename']
        return

    run['scaleTrainCommand'] = "%s -s %s %s > %s" % (libsvmScalePath, run['scaleParamsFilename'], run['extractTrainFilename'], run['scaleTrainFilename'])
    startTime = time.time()
    run['scaleTrainOutput'] = commands.getoutput(run['scaleTrainCommand'])
    run['scaleTrainTime'] = time.time() - startTime

    run['scaleTestCommand'] = "%s -r %s %s > %s" % (libsvmScalePath, run['scaleParamsFilename'], run['extractTestFilename'], run['scaleTestFilename'])
    startTime = time.time()
    run['scaleTestOutput'] = commands.getoutput(run['scaleTestCommand'])
    run['scaleTestTime'] = time.time() - startTime
    
    if DEBUG:
        print "scaleTrainCommand=%s" % (run['scaleTrainCommand'])
        print "scaleTrainOutput=%s" % (run['scaleTrainOutput'])
        print "scaleTrainTime=%s" % (run['scaleTrainTime'])
        print "scaleTestCommand=%s" % (run['scaleTestCommand'])
        print "scaleTestOutput=%s" % (run['scaleTestOutput'])
        print "scaleTestTime=%s" % (run['scaleTestTime'])

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

    run['trainCommand'] = "%s %s %s %s" % (trainPath, run['svmOptions'], run['scaleTrainFilename'], run['modelFilename'])
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
        run['predictCommand'] = "%s %s -t %s" % (predictPath, run['svmOptions'], run['arffTestFilename'])
    else:
        run['predictCommand'] = "%s %s %s %s" % (predictPath, run['scaleTestFilename'], run['modelFilename'], run['predictionFilename'])

    startTime = time.time()
    run['predictOutput'] = commands.getoutput(run['predictCommand'])
    run['predictTime'] = time.time() - startTime

    if weka == True:
        m = re.search('Stratified cross-validation ===\s+Correctly Classified Instances\s+[0-9]+\s+([0-9.]+)', run['predictOutput'])
        run['predictAccuracy'] = float(m.group(1))
    else:
        run['predictAccuracy'] = run['predictOutput']
    

    if DEBUG:
        print "predictCommand=%s" % (run['predictCommand'])
        print "predictOutput=%s" % (run['predictOutput'])
        print "predictTime=%s" % (run['predictTime'])
    

def removeTmpFiles(run):

    if os.path.exists(run['extractTrainFilename']):
        os.remove(run['extractTrainFilename'])
        
    if os.path.exists(run['extractTestFilename']):
        os.remove(run['extractTestFilename'])

    if os.path.exists(run['scaleTrainFilename']):
        os.remove(run['scaleTrainFilename'])
        
    if os.path.exists(run['scaleTestFilename']):
        os.remove(run['scaleTestFilename'])

    if os.path.exists(run['modelFilename']):
        os.remove(run['modelFilename'])

    if os.path.exists(run['predictionFilename']):
        os.remove(run['predictionFilename'])
        
    
def run(runs,inTrainCollection,inTestCollection):
    if DEBUG:
        print "TOOLSDIR=%s" % (TOOLSDIR)
        print "TMPDIR=%s" % (TMPDIR)
        
    for run in runs:
        runExtract(run,inTrainCollection,inTestCollection)
        runScale(run)
        runTrain(run)
        runPredict(run)
        print "|%s|%s|%s|%.2f|%.2f|%.2f|%s|" % (run['table'], run['extractOptions'], run['svmOptions'], run['extractTrainTime'] + run['extractTestTime'], run['trainTime'], run['predictTime'], run['predictAccuracy'])
        #removeTmpFiles(run)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: run-nextract-svm.py input/runsvm.test.json input/train.mf input/test.mf"
        sys.exit(1)

    inCommandFilename = sys.argv[1]
    inTrainCollection = sys.argv[2]
    inTestCollection = sys.argv[3]
    runs = parseInput(inCommandFilename)
    runs = generateFilenames(runs)
    run(runs,inTrainCollection,inTestCollection)
        

