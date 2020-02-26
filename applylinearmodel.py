#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse, sys, pickle
from sklearn.linear_model import *


def rms(x):
    return np.sqrt(np.mean(x**2))
    
def scoremodel(model, x, y):
    '''Return fitness of model. We'll use R^2 and RMS.
       We also compare to the RMS with the mean.'''
    p = model.predict(x).squeeze()
    r = rms(p-y)
    aver = rms(y-np.mean(y))  #RMS if we just used average
    return np.corrcoef(p,y)[0,1]**2,r,aver



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply linear model to fingerprint file')
    parser.add_argument('model',help='Model file')
    parser.add_argument('input',help='Fingerprints input file')
    
    args = parser.parse_args()
    
    comp = 'gzip' if args.input.endswith('.gz') else None
    data = pd.read_csv(args.input,compression=comp,header=None,delim_whitespace=True)    
    x = np.array(data.iloc[:,2:],dtype=np.float)
    smi = data.iloc[:,0]

    model = pickle.load(open(args.model))
    p = model.predict(x).squeeze()
    
    for (m, score) in zip(smi,p):
        print(m,score)

