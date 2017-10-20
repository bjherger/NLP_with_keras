# Moby-Dick

Exploration into NLP modeling w/ Keras. 

There aren't great batteries included examples for modeling text with deep learning, so I've built out this repo to 
contain starter code for:
 
 - Text processing: Processing text to be utilized with keras (text pre-processing, converting to indices, padding)
 - Pre-trained embedding: Using a pre-trained text embedding (GoogleNews 300) with keras (translating words to a point in \mathbb{R}^{300})
 - Convolutional architecture: Modeling text with a convolutional architecture (functionally similar to Ngrams)
 - RNN architecture: Modeling text with a Recurrent Neural Net (RNN) architecture (functionally similar to a rolling 
 window)
 

## Quick start
  
To run the Python code, complete the following:
```bash

# Create python virtual environment
conda env create -f environment.yml 

# Activate python virtual environment
source activate moby-dick

# Run script
cd bin/
python main.py

# Warning: The first time you run this script, it'll have to download the data set and pre-trained embeddings. This 
# will take ~15 minutes with a decent internet connection.  
```

## Getting started

### Repo structure

 - Code entry point: `bin/main.py`
 - Configuration file: `conf/confs.yaml` (you need to create it from the template!)
 - Schemas: `data/schema/[step_name].csv`

### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use 
environment from file*, and *Change environments (activate/deactivate)*). 


### Confs

This application requires some configuration before it can run correctly. Please use the commands below to set up the 
configs:

```bash
# Create configuration file
cp conf/confs.yaml.template conf/confs.yaml

# Fill out confs (This requires work on your end!)
open conf/confs.yaml
```

## Contact
Feel free to contact me at Brendan <dot> Herger <at> capitalone <dot> com
