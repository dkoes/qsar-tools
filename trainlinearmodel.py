#!/usr/bin/env python

import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train linear model from fingerprint file')
    parser.add_argument('input',help='Fingerprints input file')
    parser.add_argument('-o','--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    
    args = parser.parse_args()
    out = args.outfile
    
    comp = 'gzip' if args.input.endswith('.gz') else None
    data = pd.read_csv(args.input,compression=comp,header=None,delim_whitespace=True)
    
    smi = np.array(data.iloc[:,0])
    y = np.array(data.iloc[:,1],dtype=np.float)
    x = np.array(data.iloc[:,2:],dtype=np.float)
    del data #dispose of pandas copy