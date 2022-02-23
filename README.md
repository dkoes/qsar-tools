# qsar-tools
Scripts for assisting in modeling quantitative structure activity relationships from 2D chemical data


## createsmartsdescriptors.py

Given a training set, compute a set of smarts descriptors unique to that set.
Can choose between path-based descriptors and circular descriptors.

## outputfingerprints.py

Given a SMILES file, outputs a space delimited file of the fingerprint descriptors
for each molecule for use with training a model 

## trainlinearmodel.py

Given a fingerprint file train a linear model to predict
a numerical quantity of interest.  This quantity is assumed to be the second
column of the file.  Outputs model along with cross-validation statistics.

## applylinearmodel.py

Given a model and a set of compounds, uses the model to predict the quantity 
of interest.

## trainclassifier.py

Given a fingerprint file train a classifier model to predict
a the class of interest. Does NOT support multiple classes.
This quantity is assumed to be the second column of the file.
Outputs model along with cross-validation statistics.

## applyclassifier.py

Given a model and a set of compounds, uses the model to predict the class 
of interest.

## visualizesmartsmodel.py

Given a model trained using smarts descriptors and a compound, map the model
onto the compound and produce an image.

# Example

`data/solubility.smi` contains SMILES where the molecule name is the desired numerical label (logS in this case).

```
$ time ./createsmartsdescriptors.py --subgraph -c 3 data/solubility.smi -o solubility_smarts.txt
real	0m48.485s
user	0m48.611s
sys	0m0.370s

$ wc solubility_smarts.txt 
  8947  17894 161131 solubility_smarts.txt

$ time ./outputfingerprints.py --smarts --smartsfile solubility_smarts.txt data/solubility.smi -o solubility.fp.gz
real	0m46.029s
user	0m46.247s
sys	0m0.589s

$ time ./trainlinearmodel.py --elastic solubility.fp.gz -o solubility.model
CV: R^2=0.7765, RMS=0.9836, NullRMS=2.0738 (stds 0.0081, 0.0188, 0.0070)
Gap: R^2=0.1205, RMS=-0.3087, NullRMS=0.0000

real	89m29.045s
user	1504m2.279s
sys	515m52.762s

# the following will make images named using the molecule names in one.smi
$ ./visualizesmartsmodel.py solubility.model solubility_smarts.txt one.smi
```
