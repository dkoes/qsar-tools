#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, sys, pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def rms(x):
    return np.sqrt(np.mean(x**2))
    
def scoremodel(model, x, y):
    '''Return fitness of model. We'll use R^2 and RMS.
       We also compare to the RMS with the mean.'''
    p = model.predict(x).squeeze()
    r = rms(p-y)
    aver = rms(y-np.mean(y))  #RMS if we just used average
    if np.std(p) == 0.0 or np.std(y) == 0: #R not defined
        return 0,r,aver
        
    return np.corrcoef(p,y)[0][1]**2,r,aver

def trainmodels(m, x, y, iter=1000):
    '''For the model type m, train a model on x->y using built-in CV to
    parameterize.  Return both this model and an unfit model that can be used for CV.
    Note for PLS we cheat a little bit since there isn't a built-in CV trainer.
    '''
    
    if m == 'pls':
        #have to manually cross-validate to choose number of components
        kf = KFold(n_splits=3)
        bestscore = -10000
        besti = 0
        for i in range(1,min(100,len(x[0]))):
            #try larger number of components until average CV perf decreases
            pls = PLSRegression(i)
            scores = []
            #TODO: parallelize below
            for train,test in kf.split(x):
                xtrain = x[train]
                ytrain = y[train]
                xtest = x[test]
                ytest = y[test]            
                pls.fit(xtrain,ytrain)
                score = scoremodel(pls,xtest,ytest)
                scores.append(score)
                
            ave = np.mean(scores)
            if ave < bestscore*0.95: #getting significantly worse
                break
            elif ave > bestscore:
                bestscore = ave
                besti = i
        
        model = PLSRegression(besti) 
        model.fit(x,y)
        unfit = PLSRegression(besti)  #choose number of components using full data - iffy
        print("PLS components =",besti)

    elif m == 'lasso':
        model = LassoCV(n_jobs=-1,max_iter=iter)
        model.fit(x,y)
        unfit = LassoCV(n_jobs=-1,max_iter=iter) #(alpha=model.alpha_)
        print ("LASSO alpha =",model.alpha_)
        return (model,unfit)
    elif m == 'ridge':
        model = RidgeCV()
        model.fit(x,y)
        print ("Ridge alpha =",model.alpha_)
        unfit = RidgeCV()
    else:
        model = ElasticNetCV(n_jobs=-1,l1_ratio=[.1, .5, .7, .9, .95, .99, 1],max_iter=iter)
        model.fit(x,y)
        print("Elastic alpha =",model.alpha_," l1_ratio =",model.l1_ratio_)
        unfit = ElasticNetCV(n_jobs=-1,max_iter=iter)

    return (model,unfit)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train linear model from fingerprint file')
    parser.add_argument('input',help='Fingerprints input file')
    parser.add_argument('-o','--outfile', type=argparse.FileType('wb'), help="Output file for model (trained on full data)")
    parser.add_argument('-k','--kfolds',type=int,default=3,help="Number of folds for cross-validation")
    parser.add_argument('-y','--affinities',help="Affinities (y-values). Will override any specified in fingerprints file")
    parser.add_argument('--maxiter',type=int,help="Maximum number of iterations for iterative methods.",default=1000)
    
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--lasso',action='store_const',dest='model',const='lasso',help="Use LASSO linear model")
    models.add_argument('--elastic',action='store_const',dest='model',const='elastic',help="Use ElasticNet linear model")
    models.add_argument('--ridge',action='store_const',dest='model',const='ridge',help="Use Ridge linear model")
    models.add_argument('--pls',action='store_const',dest='model',const='pls',help="Use Partial Least Squares")

    parser.set_defaults(model='lasso')
    
    args = parser.parse_args()
    #out = args.outfile
    
    comp = 'gzip' if args.input.endswith('.gz') else None
    data = pd.read_csv(args.input,compression=comp,header=None,delim_whitespace=True)
    
    if args.affinities: #override what is in fingerprint file
        y = np.genfromtxt(args.affinities,np.float)
        if len(y) != len(data):
            print ("Mismatched length between affinities and fingerprints (%d vs %d)" % (len(y),len(x)))
            sys.exit(-1)
        data.iloc[:,1] = y
    
    np.random.seed(0) #I like reproducible results, so fix a seed
    data = data.iloc[np.random.permutation(len(data))] #shuffle order of data
    smi = np.array(data.iloc[:,0])
    

    y = np.array(data.iloc[:,1],dtype=float)
    x = np.array(data.iloc[:,2:],dtype=float)
    del data #dispose of pandas copy    
    
    (fit,unfit) = trainmodels(args.model, x, y, args.maxiter)
    fitscore = scoremodel(fit,x,y)
    print ("Full Regression: R^2=%.4f, RMS=%.4f, NullRMS=%.4f" % fitscore)
    nz = np.count_nonzero(fit.coef_)
    print("Nonzeros: %d (%.2f%%)" % (nz,100.0*nz/float(len(fit.coef_))))
    kf = KFold(n_splits=args.kfolds)
    scores = []
    for train,test in kf.split(x):
        xtrain = x[train]
        ytrain = y[train]
        xtest = x[test]
        ytest = y[test]        
        unfit.fit(xtrain,ytrain)
        scores.append(scoremodel(unfit, xtest, ytest))
        
    print(f"{args.kfolds}-fold CV: R^2=%.4f, RMS=%.4f, NullRMS=%.4f (stds %.4f, %.4f, %.4f)" % (tuple(np.mean(scores, axis=0)) + tuple(np.std(scores,axis=0))))
    print("Gap: R^2=%.4f, RMS=%.4f, NullRMS=%.4f" % tuple(fitscore-np.mean(scores,axis=0)))
            
    if args.outfile:
        pickle.dump(fit, args.outfile )
