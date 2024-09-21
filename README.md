# EVOLVE: A webserver to explore viral evolution to better distinguish a "Variant Under Monitoring (VUM)" from multiple sequence data learning
Satyam Sangeet<sup>12</sup>, Anushree Sinha<sup>1</sup>, Madhav B. Nair<sup>1</sup>, Arpita Mahata<sup>1</sup>, Raju Sarkar<sup>1</sup>, Susmita Roy<sup>1</sup>
- <sup>1</sup>Department of Chemical Sciences, Indian Institute of Science Education and Research, Kolkata, India
- <sup>2</sup>School of Physics, The University of Sydney, NSW, Australia

<img src="logo.jpg"  alt="Evolve Logo" height="200">

***
Welcome to the EVOLVE repository! This repository contains the relevant data related to teh EVOLVE webserver. Below is the architecture of the repository, along with a description of the different directories and their contents.

# Local Calculation
- This folder contains Python code that performs specific analyses on local machines. These analyses are computationally expensive or require resources that are not available on the web server. 
- The scripts here are optimized for local execution.
- The python scripts are annotated to provide instructions in case of any doubt.

# Models
This folder contains all the models that have been used or developed during the project. It is divided into two subfolders:
- Pre-trained models: This contains four models that have been pre-trained on specific datasets:
  - Omicron: A model trained for the Omicron variant.
  - Influenza: A model trained for Influenza virus data.
  - CTV: A model focused on Citrus Tristeza Virus (CTV).
  - Xanthomonas: A model for the Xanthomonas genus of bacteria.

- Transfer learned models: This subfolder contains models that have been further trained (transfer learning) on specific datasets. Currently, it includes:
  - Dengue Virus Model: A model fine-tuned for detecting or predicting Dengue virus characteristics using transfer learning techniques.

More details are provided in the respetive directory

# Supplementary
- This folder stores the results of various tests and model training experiments.
- These results correspond to different test and train data splits, showcasing the performance and accuracy of the models over a range of training sizes.
- It includes performance metrics, validation results, and any additional supplementary information.

# Training Data
- This folder holds the training datasets used for the models listed above.
- Each model has a corresponding dataset, which has been used to train the model from scratch or fine-tune it during the transfer learning process.

***
