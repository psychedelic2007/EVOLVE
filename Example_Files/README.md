# Example Files for Webserver

## Overview
This repository contains example files that can be used to test the functionalities of a webserver designed for sequence analysis. The files are categorized based on their purpose and correspond to different sections of the server.

## Directory Structure
```
📂 example_files
   ├── entropy1.aln
   ├── entropy2.aln
   ├── example_mrf.csv
   ├── example_prediction.csv
   ├── raw_sequences.fasta
```
Each file is designed for a specific task in the webserver and can be uploaded to the corresponding tabs.

## File Descriptions and Usage

### 1. **Entropy Calculation Files**
**Files:**
- `entropy1.aln`
- `entropy2.aln`

These files contain multiple sequence alignments and should be uploaded to the **Entropy** tab of the webserver. The server will calculate and plot the entropy of aligned sequences, helping in understanding sequence variability.

### 2. **Mutational Response Function (MRF) Calculation File**
**File:** `example_mrf.csv`

This file should be uploaded to the **MRF** tab to compute the Mutational Response Function. The MRF helps in understanding how mutations affect the functional stability of the sequence.

### 3. **Mutation Prediction File**
**File:** `example_prediction.csv`

This file should be uploaded to the **Prediction** tab to analyze potential mutations. The webserver will use predictive models to estimate the impact of mutations on the sequence.

### 4. **Sequence Preprocessing File**
**File:** `raw_sequences.fasta`

This FASTA file contains raw sequences and should be uploaded to the **Preprocessing** tab of the webserver. The preprocessing step includes sequence cleaning, trimming, and formatting for further analysis.

## How to Use
1. Download the repository:
   ```bash
   git clone https://github.com/yourusername/example_files.git
   cd example_files
   ```
2. Navigate to the webserver and go to the corresponding tab:
   - Upload `entropy1.aln` or `entropy2.aln` in the **Entropy** tab.
   - Upload `example_mrf.csv` in the **MRF** tab.
   - Upload `example_prediction.csv` in the **Prediction** tab.
   - Upload `raw_sequences.fasta` in the **Preprocessing** tab.
3. Run the analysis and interpret the results.

## Contributions
Contributions to this repository are welcome! If you have additional example files or improvements, feel free to submit a pull request.

## License
This repository is available under [MIT License](LICENSE).

## Contact
For any inquiries, please reach out via [webserver.iiserkol@gmail.com] or create an issue in this repository.
