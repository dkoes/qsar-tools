#!/usr/bin/env python3

'''Routines for enumerating SMARTS expressions representing fragments of provided molecules'''

import sys,argparse,collections
import numpy as np
from rdkit.Chem import AllChem as Chem

def computepathsmarts(mol, size):
    if size <= 0: size = 7 # default to 7 atoms
    ret = set()
    for length in range(2,size+1):
        paths = Chem.FindAllPathsOfLengthN(mol, length, useBonds=False)
        for path in paths:
            smi = Chem.MolFragmentToSmiles(mol, path, canonical=True, allBondsExplicit=True)
            ret.add(smi)
    return ret

def computesubgraphsmarts(mol, size):
    '''Given an rdkit mol, extract all the paths up to length size (if zero,
    use default length).  Return smarts expressions for these paths.'''
    if size <= 0: size = 6 #default to 6 bonds
    #find all paths of various sizes
    paths = Chem.FindAllSubgraphsOfLengthMToN(mol,1,size)
    #paths is a list of lists, each of which is the bond indices for paths of a different length
    paths = [path for subpath in paths for path in subpath]
    ret = set()
    for path in paths:
        atoms=set()
        for bidx in path:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        smi = Chem.MolFragmentToSmiles(mol, atoms, canonical=True, allBondsExplicit=True)
        ret.add(smi)
    return ret

def computecircularsmarts(mol, size):
    '''Given an rdkit mol, extract all the circular type descriptors up to 
    radius size (if zero, use default length).  Return smarts expressions for these fragments.'''
    if size <= 0: size = 2
    ret = set()
    for a in mol.GetAtoms():
        for r in range(1,size+1):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol,r,a.GetIdx())
            atoms=set()
            for bidx in env:
                atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
                atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
            smi = Chem.MolFragmentToSmiles(mol, atoms, canonical=True, allBondsExplicit=True)
            ret.add(smi)
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create SMARTS-based descriptors from set of molecules')
    parser.add_argument('smi',help='SMILES input file')
    parser.add_argument('-o','--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    
    descriptortype = parser.add_mutually_exclusive_group()
    descriptortype.add_argument('--path',action='store_const',dest="type",const="path",help='Use path descriptors')
    descriptortype.add_argument('--circular',action='store_const',dest="type",const="circular",help='Use circular descriptors')
    descriptortype.add_argument('--subgraph',action='store_const',dest="type",const="subgraph",help='Use subgraph descriptors')
    parser.set_defaults(type='path')
    
    parser.add_argument('--size',type=int,default=0,help="Max descriptor size (path length or circular radius)")
    parser.add_argument('-c','--cutoff',type=int,metavar="N",default=0,help="Remove smarts that appear in <= N molecules")
 
    args = parser.parse_args()

    smartcnts = collections.defaultdict(int) #count the number of molecules that have a given smart
    with open(args.smi) as f:
        for line in f:
            try:
                if line.startswith('#') or len(line.strip()) == 0:
                    continue
                else:
                    tokens = line.split()
                    mol = Chem.MolFromSmiles(tokens[0])
                    if args.type == 'path':
                        smarts = computepathsmarts(mol, args.size)
                    elif args.type == 'circular':
                        smarts = computecircularsmarts(mol, args.size)
                    elif args.type == 'subgraph':
                        smarts = computesubgraphsmarts(mol, args.size)
                    else: #should never happen
                        smarts = []
                    
                    for s in smarts:
                        smartcnts[s] += 1
            except Exception as e:
                sys.stderr.write("%s\nProblem with line: %s" % (e,line))
    
    for (smart,cnt) in smartcnts.items():
        if cnt > args.cutoff:
            args.outfile.write('%s %d\n' % (smart,cnt))
