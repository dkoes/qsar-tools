#!/usr/bin/env python

import numpy as np
import argparse, sys, pickle, math, rdkit, matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from rdkit.Chem import AllChem as Chem
import matplotlib.pylab as plt
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib.colors import LinearSegmentedColormap
import matlab
import warnings

warnings.simplefilter('ignore',FutureWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visually map model trained using smarts patterns onto molecule. Will output similarity maps.')
    parser.add_argument('model',type=argparse.FileType('r'),help='Model file')
    parser.add_argument('smarts',type=argparse.FileType('r'),help='SMARTS file')
    parser.add_argument('smi',type=argparse.FileType('r'),help='SMILES file to annotate')
    parser.add_argument('--size',type=int,default=250,help='Image size')
    parser.add_argument('--labels',action='store_true',default=False,help='Show labels in image')
     
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
        
    #process each cmpd, save weights and mol - want to
    mols = []
    weights = []
    maxval = 0
    for (i,line) in enumerate(args.smi):
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        tokens = line.split()
        mol = Chem.MolFromSmiles(tokens[0])
        if len(tokens) < 2:
            name = str(i)
        else:
            name = tokens[1]
        
        mol.title = name
        mols.append(mol)
        #apply weights 
        atomcontribs = np.zeros(mol.GetNumAtoms())
        for c in nonzeroes:
            pat = smarts[c];
            w = coefs[c]/float(pat.GetNumAtoms())
            matches = mol.GetSubstructMatches(pat, True)
            for m in matches: #each match is a tuple of atom indices
                for a in m:
                    atomcontribs[a] += w
                    
        weights.append(atomcontribs)
        maxval = max(maxval,np.max(np.abs(atomcontribs)))
        
    #normalize the weights
    weights = [w/maxval for w in weights]
    
    #make a custom red - white - green color map
    cdict = {'red': ( (0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.0, 0.0)),
             'green': ( (0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),
             'blue': ( (0.0, 0.0, 0.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.0, 0.0))};
    colors = LinearSegmentedColormap('RWG',cdict)
    
   
    size = (args.size,args.size) #get errors if not square
    #this is an awfully hacky way to standardize the color ranges across
    #all the processed molecules. A major issue is that scale and sigma
    #are in image coordinates
    sigma = 0.05
    if len(mols) and mols[0].GetNumBonds() > 0:
        mol = mols[0]
        rdkit.Chem.Draw.MolToMPL(mol,size=size) #initialize 2D coordinates
        plt.clf()
        bond = mol.GetBondWithIdx(0) 
        idx1 = bond.GetBeginAtomIdx() 
        idx2 = bond.GetEndAtomIdx() 
        sigma = 0.5 * math.sqrt(sum([(mol._atomPs[idx1][i]-mol._atomPs[idx2][i])**2 for i in range(2)]))
        
    scale = matlab.bivariate_normal(0,0,sigma,sigma,0,0)
    
    for (mol,w) in zip(mols,weights):        
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, w, size=size, scale=scale,sigma=sigma,colorMap=colors,contourLines=1,alpha=0) #alpha hides contour lines
        
        if not args.labels:
            #I don't like white boxes on labels
            for elem in fig.axes[0].get_children():
                if isinstance(elem, matplotlib.text.Text):
                    elem.set_visible(False)

        plt.axis("off")
        plt.savefig('%s.png' % mol.title,bbox_inches='tight')

        