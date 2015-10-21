#!/usr/bin/env python

'''Routines for creating count and binary fingerprints from molecules'''

import sys,argparse,collections,re,gzip
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MACCSkeys

def loadsmarts(fname):
    ret = []
    with open(fname) as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            tokens = line.split()
            mol = Chem.MolFromSmarts(tokens[0])
            ret.append(mol)
    return ret

def addfpargs(parser):
    '''add options needed to customize fingerprint'''
    fp = parser.add_mutually_exclusive_group()
    fp.add_argument('--rdkit',action='store_const',dest='fp',const='rdkit',help="Use default RDKit fingerprints")
    fp.add_argument('--ecfp4',action='store_const',dest='fp',const='ecfp4',help="Use ECFP4 fingerprints")
    fp.add_argument('--ecfp6',action='store_const',dest='fp',const='ecfp6',help="Use ECFP6 fingerprints")
    fp.add_argument('--maccs',action='store_const',dest='fp',const='maccs',help="Use MACCS fingerprints")
    fp.add_argument('--smarts',action='store_const',dest='fp',const='smarts',help="Use SMARTS fingerprints (must provide file)")
    parser.set_defaults(fp='rdkit')

    parser.add_argument('--fpbits',type=int,default=2048,help="Number of bits to use in folded fingerprints")
    parser.add_argument('--smartsfile',type=loadsmarts,help="File SMARTS to use for smarts fingerprints")


def calcfingerprint(mol, args):
    '''Return a list of the bits/cnts of the fingerprint of mol.
    Uses fingerprint settings from args which should have been the result
    of a parse of addfpargs'''
    if args.fp == 'rdkit':
        fp = Chem.RDKFingerprint(mol,fpSize=args.fpbits)
        return [int(x) for x in fp.ToBitString()]
    elif args.fp.startswith('ecfp'):
        diameter = int(args.fp.replace('ecfp',''))
        r = diameter/2
        fp = Chem.GetMorganFingerprintAsBitVect(mol,r,nBits=args.fpbits)
        return [int(x) for x in fp.ToBitString()]
    elif args.fp == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
        return [int(x) for x in fp.ToBitString()]
    elif args.fp == 'smarts':
        if args.smartsfile:
            smarts = args.smartsfile
            ret = [0]*len(smarts)
            for (i,smart) in enumerate(smarts):
                if mol.HasSubstructMatch(smart):
                    ret[i] = 1
            return ret
        else:
            sys.stderr.write("ERROR: Must provide SMARTS file with --smarts\n")
            sys.exit(-1)
    else:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Output fingerprints in plain text')
    parser.add_argument('smi',help='SMILES input file')
    parser.add_argument('-o','--outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    
    addfpargs(parser)
    
    args = parser.parse_args()
    
    #support gzipped output - these files are big
    if args.outfile.name.endswith('.gz'):
        out = gzip.GzipFile(fileobj=args.outfile)
    else:
        out = args.outfile
        
    with open(args.smi) as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue
            tokens = line.split()
            mol = Chem.MolFromSmiles(tokens[0])
            fp = calcfingerprint(mol, args)
            #we define the output to have exactly two columns before the bits
            #hopefully the last thing is the affinity?
            aff = tokens[-1]
            if len(tokens) == 1: aff = "_"  #if no affinity, blank
            out.write('%s %s %s\n' % (tokens[0],aff,' '.join(map(str,fp))))
