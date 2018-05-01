#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse, sys, pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from sklearn.model_selection import KFold
from sklearn.metrics import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
    
def scoremodel(model, x, y):
    '''Return fitness of model. We'll use AUC.'''
    p = model.predict(x).squeeze()
    return roc_auc_score(y,p)


def trainmodels(m, x, y):
    '''For the model type m, train a classifier on x->y using built-in CV to
    parameterize.  Return both this model and an unfit model that can be used for CV.
    Note for PLS we cheat a little bit since there isn't a built-in CV trainer.
    '''
    
    if m == 'knn':
        #have to manually cross-validate to choose number of components
        kf = KFold(n_splits=3)
        bestscore = -10000
        besti = 0
        for i in xrange(1,10):
            #try larger number of components until average CV perf decreases
            knn = KNeighborsClassifier(i)
            scores = []
            #TODO: parallelize below
            for train,test in kf.split(x):
                xtrain = x[train]
                ytrain = y[train]
                xtest = x[test]
                ytest = y[test]            
                knn.fit(xtrain,ytrain)
                score = scoremodel(knn,xtest,ytest)
                scores.append(score)
                
            ave = np.mean(scores)
            if ave > bestscore:
                bestscore = ave
                besti = i
        
        model = KNeighborsClassifier(besti) 
        model.fit(x,y)
        unfit = KNeighborsClassifier(besti)  #choose number of components using full data - iffy
    elif m == 'svm':
        #TODO: paramter optimization
        model = svm.SVC()
        model.fit(x,y)
        unfit = svm.SVC() 
    elif m == 'logistic':
        model = LogisticRegressionCV(n_jobs=-1)
        model.fit(x,y)
        unfit = LogisticRegressionCV(n_jobs=-1)
    elif m == 'rf':
        #TODO: samae
        model = RandomForestClassifier()
        model.fit(x,y)
        unfit = RandomForestClassifier()


    return (model,unfit)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train linear model from fingerprint file')
    parser.add_argument('input',help='Fingerprints input file')
    parser.add_argument('-o','--outfile', type=argparse.FileType('w'), help="Output file for model (trained on full data)")
    parser.add_argument('-k','--kfolds',type=int,default=3,help="Number of folds for cross-validation")
    parser.add_argument('-y','--labels',help="Labels (y-values). Will override any specified in fingerprints file")
    
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--svm',action='store_const',dest='model',const='svm',help="Use support vector machine (rbf kernel)")
    models.add_argument('--knn',action='store_const',dest='model',const='knn',help="Use k-nearest neighbors")
    models.add_argument('--rf',action='store_const',dest='model',const='rf',help="Use random forest")
    models.add_argument('--logistic',action='store_const',dest='model',const='logistic',help="Use logistic regression")

    parser.set_defaults(model='knn')
    
    args = parser.parse_args()
    out = args.outfile
    
    comp = 'gzip' if args.input.endswith('.gz') else None
    data = pd.read_csv(args.input,compression=comp,header=None,delim_whitespace=True)
    
    if args.labels: #override what is in fingerprint file
        y = np.genfromtxt(args.labels,np.float)
        if len(y) != len(data):
            print "Mismatched length between affinities and fingerprints (%d vs %d)" % (len(y),len(x))
            sys.exit(-1)
        data.iloc[:,1] = y
    
    np.random.seed(0) #I like reproducible results, so fix a seed
    data = data.iloc[np.random.permutation(len(data))] #shuffle order of data
    smi = np.array(data.iloc[:,0])
    

    y = np.array(data.iloc[:,1],dtype=np.float)
    x = np.array(data.iloc[:,2:],dtype=np.float)
    del data #dispose of pandas copy    
    
    (fit,unfit) = trainmodels(args.model, x, y, args.maxiter)
    fitscore = scoremodel(fit,x,y)
    print "Full Regression: AUC=%.4f" % fitscore

    kf = KFold(n_splits=3)
    scores = []
    for train,test in kf.split(x):
        xtrain = x[train]
        ytrain = y[train]
        xtest = x[test]
        ytest = y[test]        
        unfit.fit(xtrain,ytrain)
        scores.append(scoremodel(unfit, xtest, ytest))
        
    print "CV: AUC=%.4f (std %.4f)" % (np.mean(scores), np.std(scores))
    print "Gap: %.4f" % fitscore-np.mean(scores)
            
    if args.outfile:
        pickle.dump(fit, args.outfile, pickle.HIGHEST_PROTOCOL)
