# Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications

# Introduction
This is Kalen Wei's testing repository based on Prithri 2.0.

Prithri 2.0 is a **Fundational Model** pretrained for Earth Observation Applications, which is mainly a ViT-based model.

Hence, this repository is mainly for testing the Prithri 2.0 model by multiple manipulations. 

# Usage
## Installation

I'd recommend you to create a new conda environment for this project.

`conda create -n prithvi python=3.11`

Activate the environment.

`conda activate prithvi`

Then install the required packages. Please note that **DO NOT FOLLOW THE OFFICIAL INSTRUCTIONS!**

Just:

`pip install terratorch==0.99.8`

and nothing more.



## Running testing scripts

1. get into conda environment
`conda activate prithvi` if you created the environment before.
2. run the testing script `python run.py`,`python run_moe.py`, `python run_itransformer.py` based on what you need.

3. check the results
the results will be saved in `finetuned_checkpoints` folder.
