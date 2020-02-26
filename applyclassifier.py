#!/usr/bin/env python3
'''Apply a classification model trained with trainclassifier.py'''
import numpy as np
import pandas as pd
import argparse, sys, pickle
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
    

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

    y = np.array(data.iloc[:,1],dtype=np.float)
    print("AUC = %.4f"%roc_auc_score(y,p))

