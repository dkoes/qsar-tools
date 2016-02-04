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

## visualizesmartsmodel.py

Given a model trained using smarts descriptors and a compound, map the model
onto the compound and produce an image.
