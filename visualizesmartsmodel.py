#!/usr/bin/env python

import numpy as np
import argparse, sys, pickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from rdkit.Chem import AllChem as Chem
import matplotlib.pylab as plt
from rdkit.Chem.Draw import SimilarityMaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visually map model trained using smarts patterns onto molecule. Will output similarity maps.')
    parser.add_argument('model',type=argparse.FileType('r'),help='Model file')
    parser.add_argument('smarts',type=argparse.FileType('r'),help='SMARTS file')
    parser.add_argument('smi',type=argparse.FileType('r'),help='SMILES file to annotate')
     
    args = parser.parse_args()

    #read model
    model = pickle.load(args.model)
    
    #read Smarts
    smarts = []
    for line in args.smarts:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        tokens = line.split()
        mol = Chem.MolFromSmarts(tokens[0])
        smarts.append(mol)
        
    #get coefficients from model
    coefs = model.coef_
    nonzeroes = np.nonzero(coefs)[0]
    if len(coefs) != len(smarts):
        print "Mismatch between model size and number fo smarts (%d vs %d)" % (len(coefs), len(smarts))
        sys.exit(-1)
        
    #process each cmpd
    for (i,line) in enumerate(args.smi):
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        tokens = line.split()
        mol = Chem.MolFromSmiles(tokens[0])
        if len(tokens) < 2:
            name = str(i)
        else:
            name = tokens[1]
        
        #apply weights 
        atomcontribs = np.zeros(mol.GetNumAtoms())
        for c in nonzeroes:
            pat = smarts[c];
            w = coefs[c]/float(pat.GetNumAtoms())
            matches = mol.GetSubstructMatches(pat, True)
            for m in matches: #each match is a tuple of atom indices
                for a in m:
                    atomcontribs[a] += w
                    
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, atomcontribs)
        print atomcontribs
        plt.axis("off")

        plt.savefig('%s.png' % name,bbox_inches='tight')

        