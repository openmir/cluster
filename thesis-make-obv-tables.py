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

import pprint
pp = pprint.PrettyPrinter(indent=4)

def run():
    data = {}
    line = sys.stdin.readline()
    while line:
        a = line.split("|")
        item = {}
        item['table'] = a[1]
        item['extractOptions'] = a[2]
        item['svmOptions'] = a[3]
        item['extractTime'] = float(a[4])
        item['trainTime'] = float(a[5])
        item['predictTime'] = float(a[6])
        item['accuracy'] = float(a[7])

        key = "%s_%s_%s" % (item['table'], item['extractOptions'], item['svmOptions'])

        if key not in data:
            data[key] = {}
            data[key]['items'] = []

        data[key]['items'].append(item)
        
        line = sys.stdin.readline()


    # pp.pprint(data)

    # sys.exit(1)
    
    for n in data:
        extractTimes = []
        trainTimes = []
        predictTimes = []
        accuracies = []
        for item in data[n]['items']:
            extractTimes.append(item['extractTime'])
            trainTimes.append(item['trainTime'])
            predictTimes.append(item['predictTime'])
            accuracies.append(item['accuracy'])

        # print "accuracies="
        # print accuracies
        # print np.mean(accuracies)

        # print data[n]

        # sys.exit(0)
        
        data[n]['extractTimesMean'] = np.mean(extractTimes)
        data[n]['extractTimesStd'] = np.mean(extractTimes)
        data[n]['trainTimesMean'] = np.mean(trainTimes)
        data[n]['trainTimesStd'] = np.mean(trainTimes)
        data[n]['predictTimesMean'] = np.mean(predictTimes)
        data[n]['predictTimesStd'] = np.mean(predictTimes)

        data[n]['accuraciesMean'] = np.mean(accuracies)
        data[n]['accuraciesStd'] = np.mean(accuracies)

    for n in data:
        print "|%s|%s|%s|%.2f|%.2f|%.2f|%.2f|" % (
            data[n]['items'][0]['table'],data[n]['items'][0]['extractOptions'],data[n]['items'][0]['svmOptions'],
            data[n]['extractTimesMean'],data[n]['trainTimesMean'],data[n]['predictTimesMean'],
            data[n]['accuraciesMean'])
    
if __name__ == "__main__":
    if len(sys.argv) < 1:
        print "Usage: thesis-process-table-results.py"
        sys.exit(1)

    run()
        

