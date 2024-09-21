# EVOLVE: A webserver to explore viral evolution to better distinguish a "Variant Under Monitoring (VUM)" from multiple sequence data learning

<img src="logo.jpg"  alt="Evolve Logo" height="200">

***
Welcome to the EVOLVE repository! This repository contains the relevant data related to teh EVOLVE webserver. Below is the architecture of the repository, along with a description of the different directories and their contents.

# Local Calculation
- This folder contains Python code that performs specific analyses on local machines. These analyses are computationally expensive or require resources that are not available on the web server. 
- The scripts here are optimized for local execution.
- The python scripts are annotated to provide instructions in case of any doubt.

# Models
This folder contains all the models that have been used or developed during the project. It is divided into two subfolders:
    Pre-trained models: This contains four models that have been pre-trained on specific datasets:
Omicron: A model trained for the Omicron variant.
Influenza: A model trained for Influenza virus data.
CTV: A model focused on Citrus Tristeza Virus (CTV).
Xanthomonas: A model for the Xanthomonas genus of bacteria.

