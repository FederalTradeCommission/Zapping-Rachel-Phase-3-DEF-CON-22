#!/usr/bin/env python2.7

import csv, sys, math, datetime, json
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy

MIN_CALLS = 2

time_format = "%m/%d/%Y %H:%M"
trainingkeys = ['time_entropy','recp_entropy','time_mean','time_std','to_area_entropy','fr_area_entropy']

def main():
  trainfile = sys.argv[1]
  testfile = sys.argv[2]
  outfile = sys.argv[3]

  print '[*] reading files...'
  training = renderFile(trainfile)
  testing = renderFile(testfile)
  print '[*] processing input...'

  known_robots = []
  for row in training:
    if row[0] == True:
      known_robots.append(row[1])

  print '[+] beginning learning'
  forest = doLearning(training)
  print '[-] done learning'

  new_robots = buildRobots(testing,forest,trainingkeys)
  robots = set(known_robots+new_robots)

  print '[*] writing output to: ', outfile
  writeUpdatedCsv(testfile,robots,outfile)

  print 'DONE'

def renderFile(filename):
  infile = open(filename)

  keys = []
  keys.extend( ['robo'] )
  keys.extend( ['caller','count'] )
  keys.extend( ['time_entropy','recp_entropy','time_mean','time_std','to_area_entropy','fr_area_entropy'] )

  inread = csv.DictReader(infile)

  callers = {}

  # build caller based data maps
  # numbers are either robocallers or not based on data
  # use first instance to mark binary state
  for line in inread:
    f = line['FROM']
    t = line['TO']
    to_area = t[:-4]
    fr_area = f[:-4]
    robo = (line['LIKELY ROBOCALL'] == 'X')
    

    if f not in callers:
      callers[f] = {}
      callers[f]['caller'] = f
      callers[f]['times'] = []
      callers[f]['tos'] = {}
      callers[f]['robo'] = robo
      callers[f]['count'] = 0
      callers[f]['fr_areas'] = {}
      callers[f]['to_areas'] = {}

    callers[f]['times'].append(datetime.datetime.strptime(line['DATE/TIME'],time_format))

    if t not in callers[f]['tos']:
      callers[f]['tos'][t] = 0

    if to_area not in callers[f]['to_areas']:
      callers[f]['to_areas'][to_area] = 0

    if fr_area not in callers[f]['fr_areas']:
      callers[f]['fr_areas'][fr_area] = 0

    callers[f]['tos'][t] += 1
    callers[f]['count'] +=1
    callers[f]['to_areas'][to_area] += 1
    callers[f]['fr_areas'][fr_area] += 1


  # build delta times
  for caller in callers.keys():
    first = True
    last_time = None
    callers[caller]['deltas'] = {}
    for tm in sorted(callers[caller]['times']):
      if first == True:
        last_time = tm
        first = False
      else:
        delta = (tm - last_time).seconds
        last_time = tm
        if delta not in callers[caller]['deltas'].keys():
          callers[caller]['deltas'][delta] = 0
        callers[caller]['deltas'][delta] += 1


  # compute entropies for (times,recepients)
  # where number of calls exceeds MIN_CALLS - 1
  min_calls = MIN_CALLS - 1

  for caller in callers.keys():
    if callers[caller]['count'] > min_calls:
      callers[caller]['time_entropy'] = getEntropy(callers[caller]['deltas'])
      callers[caller]['recp_entropy'] = getEntropy(callers[caller]['tos'])
      callers[caller]['time_mean'] = getMean(callers[caller]['deltas'])
      callers[caller]['time_std'] = getStd(callers[caller]['deltas'])
      callers[caller]['to_area_entropy'] = getEntropy(callers[caller]['to_areas'])
      callers[caller]['fr_area_entropy'] = getEntropy(callers[caller]['fr_areas'])

  training_data = []
  # grab training data set
  for caller in callers.keys():
    if 'time_entropy' in callers[caller]:
      #if callers[caller]['time_entropy'] < 2:
        #print caller, callers[caller]['robo'], callers[caller]['count'], callers[caller]['time_entropy'], callers[caller]['recp_entropy']
        #outwrite.writerow(callers[caller])
        record = []
        for k in keys:
          record.append(callers[caller][k])
        training_data.append(record)

  return training_data

def doLearning(inread):
  # load training data
  X =list()
  Y=list()
  for row in inread:
    X.append(row[3:])
    Y.append(row[0])

  # set up random forest
  clf = RandomForestClassifier(n_estimators=100)

  # build sample data
  sample_x = X[0:3*len(X)/4]
  sample_y = Y[0:3*len(X)/4]
  test_x = X[3*len(X)/4:len(X)]
  test_y = Y[3*len(X)/4:len(X)]

  #build first forest
  clf = clf.fit(sample_x, sample_y)
  last_best = 0
  this_best = clf.score(test_x,test_y)
  print '     initial best: ' + str(this_best)
  best_tree = clf

  trys = 10

  while this_best > last_best:
    last_best = this_best
    scores = [0]*trys
    trees = [clf]*trys
    for i in xrange(0,trys):
      trees[i] = RandomForestClassifier(n_estimators=100,max_features=3)
      trees[i].fit(sample_x, sample_y)
      scores[i] = trees[i].score(test_x,test_y)
    this_best = max(scores)
    print '     this best:   ', this_best
    #, scores.index(this_best), scores
    best_tree = deepcopy(trees[scores.index(this_best)])

  clf = best_tree
  print '     actual recall: ', clf.score(X,Y)
  return clf

def writeUpdatedCsv(uknownFile,robots,outputFile):
  keys = ['TO','FROM','DATE/TIME','LIKELY ROBOCALL']
  testcsv = csv.DictReader(open(uknownFile))
  output = csv.DictWriter(open(outputFile,'w'),keys)

  for call in testcsv:
    if call['FROM'] in robots:
      call['LIKELY ROBOCALL'] = 'X'
    output.writerow(call)

def buildRobots(testdata,forest,trainingkeys):
  robots = []

  for caller in testdata:
    if forest.predict(caller[3:]) == True:
      robots.append(caller[1])

  return robots

# quick and dirty entropy function
def getEntropy(m):
  n = 0
  for k in m.keys():
    n += m[k]
  
  shannon_times_n = 0.0
  for k in m.keys():
    p = float(m[k])/float(n)
    shannon_times_n += float(m[k])*math.log(p)

  try:
    shannon = shannon_times_n / float(n)
    return -shannon
  except:
    print m
    exit(-1)

# quick and dirty mean
def getMean(m):
  n = 0
  t = 0
  for k in m.keys():
    n += m[k]
    t += m[k] * k

  return float(t)/float(n)

# quick and dirty standard deviation
def getStd(m):
  n = 0
  for k in m.keys():
    n += m[k]

  e = 0.0

  x_bar = getMean(m)
  for k in m.keys():
    d = k - x_bar
    e += m[k] * (d*d)

  return math.sqrt(e/float(n))



main()
