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
import pprint
pp = pprint.PrettyPrinter(indent=4)


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
        run['extractTrainFilename'] = '%s.train.features' % (run['baseFilename'])
        run['extractTestFilename'] = '%s.test.features' % (run['baseFilename'])
        run['extractTestCollectionFilename'] = '%s.test.mf' % (run['baseFilename'])
        run['arffTrainFilename'] = '%s.train.arff' % (run['baseFilename'])
        run['arffTestFilename'] = '%s.test.arff' % (run['baseFilename'])
        run['scaleTrainFilename'] = '%s.train.scaled' % (run['baseFilename'])
        run['scaleTestFilename'] = '%s.test.scaled' % (run['baseFilename'])
        run['scaleParamsFilename'] = '%s.params' % (run['baseFilename'])
        run['modelFilename'] = '%s.model' % (run['baseFilename'])
        run['predictionFilename'] = '%s.prediction' % (run['baseFilename'])

    return runs

def runGetTrainLabels(run,inTrainCollectionFilename):
    # Open inTrainCollection
    # Get unique labels from inTrainCollection
    # Sort these labels
    inTrainCollection = open(inTrainCollectionFilename, "r")
    line = inTrainCollection.readline()

    allLabels = []
    while line:
        a = line.split()

        if not a:
            break
        
        testFilename = a[0]
        testLabel = a[1]

        allLabels.append(testLabel)
        line = inTrainCollection.readline()

    uniqueLabels = set(allLabels)
    sortedLabels = sorted(uniqueLabels)

    run['sortedLabels'] = sortedLabels

def runExtractTrain(run,inTrainCollection):
    """ Extract audio features. """
    run['extractTrainCommand'] = "%s %s %s -w %s -o %s" % (
        nextractPath, run['extractOptions'], inTrainCollection, run['arffTrainFilename'], run['extractTrainFilename'])
    startTime = time.time()    
    run['extractTrainOutput'] = commands.getoutput(run['extractTrainCommand'])
    run['extractTrainTime'] = time.time() - startTime

    if DEBUG:
        print "extractTrainCommand=%s" % (run['extractTrainCommand'])
        print "extractTrainOutput=%s" % (run['extractTrainOutput'])
        print "extractTrainTime=%s" % (run['extractTrainTime'])

def runScaleTrain(run):
    """ Scale the data with libsvm/scale. """
    if DEBUG:
        print "runScale"
    if run['scale'] == 'false':
        if DEBUG:
            print "Not scaling data"
        run['scaleTrainFilename'] = run['extractTrainFilename']
        return

    run['scaleTrainCommand'] = "%s -s %s %s > %s" % (libsvmScalePath, run['scaleParamsFilename'], run['extractTrainFilename'], run['scaleTrainFilename'])
    startTime = time.time()
    run['scaleTrainOutput'] = commands.getoutput(run['scaleTrainCommand'])
    run['scaleTrainTime'] = time.time() - startTime

    if DEBUG:
        print "scaleTrainCommand=%s" % (run['scaleTrainCommand'])
        print "scaleTrainOutput=%s" % (run['scaleTrainOutput'])
        print "scaleTrainTime=%s" % (run['scaleTrainTime'])

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

def runExtractTest(run,testFilename,testLabel):
    # Create temporary collection file for this one file
    # testFilename
    # run['extractTestCollectionFilename']
    f = open(run['extractTestCollectionFilename'],'w')
    f.write("%s\t%s\n" % (testFilename,testLabel))
    f.close()
    
    """ Extract audio features. """
    run['extractTestCommand'] = "%s %s %s -w %s -o %s" % (
        nextractPath, run['extractOptions'], run['extractTestCollectionFilename'], run['arffTestFilename'], run['extractTestFilename'])
    startTime = time.time()    
    run['extractTestOutput'] = commands.getoutput(run['extractTestCommand'])
    run['extractTestTime'] = time.time() - startTime

    if DEBUG:
        print "runExtractTest %s" % (testFilename)
        print "extractTestCommand=%s" % (run['extractTestCommand'])
        print "extractTestOutput=%s" % (run['extractTestOutput'])
        print "extractTestTime=%s" % (run['extractTestTime'])

def runScaleTest(run):
    """ Scale the data with libsvm/scale. """
    if DEBUG:
        print "runScale"
    if run['scale'] == 'false':
        if DEBUG:
            print "Not scaling data"
        run['scaleTestFilename'] = run['extractTestFilename']
        return

    run['scaleTestCommand'] = "%s -r %s %s > %s" % (libsvmScalePath, run['scaleParamsFilename'], run['extractTestFilename'], run['scaleTestFilename'])
    startTime = time.time()
    run['scaleTestOutput'] = commands.getoutput(run['scaleTestCommand'])
    run['scaleTestTime'] = time.time() - startTime

    if DEBUG:
        print "scaleTestCommand=%s" % (run['scaleTestCommand'])
        print "scaleTestOutput=%s" % (run['scaleTestOutput'])
        print "scaleTestTime=%s" % (run['scaleTestTime'])
        
def runPredictTest(run):

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
        run['predictCommand'] = "%s %s -t %s -T %s" % (predictPath, run['svmOptions'], run['arffTrainFilename'], run['arffTestFilename'])
    else:
        run['predictCommand'] = "%s %s %s %s" % (predictPath, run['scaleTestFilename'], run['modelFilename'], run['predictionFilename'])

    startTime = time.time()
    run['predictOutput'] = commands.getoutput(run['predictCommand'])
    run['predictTime'] = time.time() - startTime

    # if weka == True:
    #     m = re.search('=== Error on test data ===\s+Correctly Classified Instances\s+[0-9]+\s+([0-9.]+)', run['predictOutput'])
    #     if m is not None:
    #         run['predictAccuracy'] = float(m.group(1))
    #     else:
    #         run['predictAccuracy'] = -1.
    # else:
    #     m = re.search('Accuracy = ([0-9.]+)', run['predictOutput'])
    #     if m is not None:
    #         run['predictAccuracy'] = float(m.group(1))
    #     else:
    #         run['predictAccuracy'] = -1.

    if DEBUG:
        print "predictCommand=%s" % (run['predictCommand'])
        print "predictOutput=%s" % (run['predictOutput'])
        print "predictTime=%s" % (run['predictTime'])


def runGetPredictAccuracy(run,testLabel):
    # print "runGetPredictAccuracy"
    # Read in prediction file
    inPrediction = open(run['predictionFilename'], "r")
    line = inPrediction.readline()

    labelCounts = {}

    while line:
        index = int(line.strip())
        label = run['sortedLabels'][index]

        # print "index=%i\tlabel=%s" % (index,label)

        if label not in labelCounts:
            labelCounts[label] = 0

        labelCounts[label] += 1

        line = inPrediction.readline()

    maxLabel = max(labelCounts, key = lambda x: labelCounts.get(x) )

    if DEBUG:
        pp.pprint(labelCounts)
        print "maxLabel=%s" % maxLabel
    
    if maxLabel == testLabel:
        labelCorrect = True
    else:
        labelCorrect = False

    if DEBUG:
        if labelCorrect:
            print "****************************** Label correct"
        else:
            print "############################## Label incorrect"

    return labelCorrect
    
    
def runPredict(run,inTestCollectionFilename):
    inTestCollection = open(inTestCollectionFilename, "r")
    line = inTestCollection.readline()
    results = {}
    total = 0
    correct = 0
    while line:
        a = line.split()

        if not a:
            break
        
        testFilename = a[0]
        testLabel = a[1]

        runExtractTest(run,testFilename,testLabel)
        runScaleTest(run)
        runPredictTest(run)
        labelCorrect = runGetPredictAccuracy(run,testLabel)

        if labelCorrect:
            correct += 1
            
        total += 1
        line = inTestCollection.readline()

    accuracy = float(correct) / float(total)
    print "|calls-%s|%s|%s|%.2f|%.2f|%.2f|%.2f|" % (run['table'], run['extractOptions'], run['svmOptions'], run['extractTrainTime'], run['trainTime'], run['predictTime'], accuracy)    
    

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
        runGetTrainLabels(run,inTrainCollection)
        runExtractTrain(run,inTrainCollection)
        runScaleTrain(run)
        runTrain(run)
        runPredict(run,inTestCollection)
    
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: run-nextract-svm-train-test.py input/runsvm.test.json input/train.mf input/test.mf"
        sys.exit(1)

    inCommandFilename = sys.argv[1]
    inTrainCollection = sys.argv[2]
    inTestCollection = sys.argv[3]
    runs = parseInput(inCommandFilename)
    runs = generateFilenames(runs)
    run(runs,inTrainCollection,inTestCollection)
        
