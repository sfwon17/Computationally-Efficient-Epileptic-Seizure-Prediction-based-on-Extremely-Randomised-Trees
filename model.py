import pdb
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import random
from sklearn.ensemble import ExtraTreesClassifier
import csv
#import matplotlib as plt
import pickle
import json

settings = json.load(open('SETTINGS.json'))
pat = settings['pat']

data = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_long_train.csv')
test = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_long_test.csv')

short_feat= list(data.columns.values)
short_feat.remove('Unnamed: 0')
short_feat.remove('File')
short_feat.remove('pat')

## clean the training data by removing nans
data.dropna(thresh=data.shape[1]-3, inplace=True)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

data.fillna(0, inplace=True)
test.fillna(0, inplace=True)

data_file = data.File.values
test_file = test.File.values

# get labels
labela=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in data_file]
labelt=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in test_file]

data['L'] = labela
test['L'] = labelt

data.sort_values(['L'], inplace=True, ascending=False)
test.sort_values(['L'], inplace=True, ascending=False)

labela = data.L.values
labelt = test.L.values

data_feat = data
test_feat = test
rows = data_feat.shape[0]

###timer
data_feat = data_feat[short_feat]
test_feat = test_feat[short_feat]

feat_names = data_feat.columns
data_feat = data_feat.values
test_feat = test_feat.values

# generate model using ExtraTrees
if pat == 1:
    clf = ExtraTreesClassifier(n_estimators=3000, random_state=0, max_depth=11, n_jobs=2)
    #lr = LogisticRegression()
    #rf = RandomForestClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='gini', min_samples_split=7)
    #lda=LinearDiscriminantAnalysis()
elif pat == 2:
    clf = ExtraTreesClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='entropy')
    #lr = LogisticRegression()
    #rf = RandomForestClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='gini', min_samples_split=7)
    #lda = LinearDiscriminantAnalysis()
elif pat == 3:
    clf = ExtraTreesClassifier(n_estimators=4500, random_state=0, max_depth=15,criterion='entropy', n_jobs=2)
    #lr = LogisticRegression()
    #rf = RandomForestClassifier(n_estimators=4500, random_state=0, max_depth=15,criterion='gini', n_jobs=2,min_samples_split=7)
    #lda = LinearDiscriminantAnalysis()
    
clf.fit(data_feat, labela)
y_pred = clf.predict_proba(test_feat)

# check hold-out set
#this_AUC = metrics.roc_auc_score(labelt, y_pred[:,1])
#print("AUC: " + str(this_AUC))

#pickle.dump(clf, open(settings['model']+'/modeldump_'+str(pat)+'_ef.pkl', 'wb'))

