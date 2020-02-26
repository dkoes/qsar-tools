#!/usr/bin/env python3

import numpy as np
import argparse, sys, pickle, math, rdkit, matplotlib
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from rdkit.Chem import AllChem as Chem
import matplotlib.pylab as plt
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colorbar
from matplotlib import cm
from numpy.random import randn
import matplotlib.patheffects as PathEffects
import warnings

warnings.simplefilter('ignore', FutureWarning)

def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.

    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visually map model trained using smarts patterns onto molecule. Will output similarity maps.')
    parser.add_argument('model', type=argparse.FileType('r'), help='Model file')
    parser.add_argument('smarts', type=argparse.FileType('r'), help='SMARTS file')
    parser.add_argument('smi', type=argparse.FileType('r'), help='SMILES file to annotate')
    parser.add_argument('--size', type=int, default=250, help='Image size')
    parser.add_argument('--labels', action='store_true', default=False, help='Show labels in image')
    parser.add_argument('--weights', action='store_true', default=False, help='Show atom weights in image')
     
    args = parser.parse_args()

    # read model
    model = pickle.load(args.model)
    
    # read Smarts
    smarts = []
    for line in args.smarts:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        tokens = line.split()
        mol = Chem.MolFromSmarts(tokens[0])
        smarts.append(mol)
        
    # get coefficients from model
    coefs = model.coef_
    nonzeroes = np.nonzero(coefs)[0]
    if len(coefs) != len(smarts):
        print "Mismatch between model size and number fo smarts (%d vs %d)" % (len(coefs), len(smarts))
        sys.exit(-1)
        
    # process each cmpd, save weights and mol - want to
    mols = []
    weights = []
    maxval = 0
    for (i, line) in enumerate(args.smi):
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
        # apply weights 
        atomcontribs = np.zeros(mol.GetNumAtoms())
        for c in nonzeroes:
            pat = smarts[c];
            w = coefs[c] / float(pat.GetNumAtoms())
            matches = mol.GetSubstructMatches(pat, True)
            for m in matches:  # each match is a tuple of atom indices
                for a in m:
                    atomcontribs[a] += w
                    
        weights.append(atomcontribs)
        maxval = max(maxval, np.max(np.abs(atomcontribs)))
	
        Chem.Compute2DCoords(mol)	

#    print index, len(annotateWeights)
        
    # normalize the weights
    normweights = [w / maxval for w in weights]
    
    # make a custom red - white - green color map
    # slight tint so O and Cl are still visible
    cdict = {'red': ((0.0, 1.0, 1.0),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.2, 0.2)),
             'green': ((0.0, 0.2, 0.2),
                      (0.5, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),
             'blue': ((0.0, 0.2, 0.2),
                      (0.5, 1.0, 1.0),
                      (1.0, 0.2, 0.2))};
    colors = LinearSegmentedColormap('RWG', cdict)
   

    size = (args.size, args.size)  # get errors if not square
    # this is an awfully hacky way to standardize the color ranges across
    # all the processed molecules. A major issue is that scale and sigma
    # are in image coordinates
    sigma = 0.05
    if len(mols) and mols[0].GetNumBonds() > 0:
        mol = mols[0]
        rdkit.Chem.Draw.MolToMPL(mol, size=size)  # initialize 2D coordinates
        plt.clf()
        bond = mol.GetBondWithIdx(0) 
        idx1 = bond.GetBeginAtomIdx() 
        idx2 = bond.GetEndAtomIdx() 
        sigma = 0.5 * math.sqrt(sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i]) ** 2 for i in range(2)]))

    scale = bivariate_normal(0, 0, sigma, sigma, 0, 0)
    
    for (mol, w, normw) in zip(mols, weights, normweights):        
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, normw, size=size, scale=scale, sigma=sigma, colorMap=colors, contourLines=1, alpha=0, coordscale=1)  # alpha hides contour lines
        
        if not args.labels:
            # I don't like white boxes on labels 
            for elem in fig.axes[0].get_children():
                if isinstance(elem, matplotlib.text.Text):
                    elem.set_visible(False)		   
		    
    	plt.axis("off")
    		
    	# this is the code that plots the weights to the compounds. 
        if args.weights:
            for at in xrange(mol.GetNumAtoms()):
                x = mol._atomPs[at][0]
                y = mol._atomPs[at][1]
                plt.text(x, y, '%.2f' % w[at], path_effects=[PathEffects.withStroke(linewidth=1, foreground="blue")])	    
    
        plt.savefig('%s.png' % mol.title, bbox_inches='tight')	       
	
