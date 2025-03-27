# Scripts Repository

## Overview
This repository contains Python scripts categorized into two main folders: **Main Calculation** and **Local Calculation**. These scripts are used in the article's pipeline for various calculations and data processing tasks.

## Directory Structure
```
📂 Scripts
   ├── 📂 Main Calculation
   │   ├── pair_probability.py
   │   ├── future_count.py
   │   ├── entropy.py
   │   ├── sequence_similarity.py
   ├── 📂 Local Calculation
   │   ├── combine_csv.py
   │   ├── combine_f1_with_f3.py
   │   ├── phylogenetics_compare.py
   │   ├── combine_f1f2_with_phylogenetics_data.py
```

## Main Calculation Scripts
These scripts perform core calculations in the article’s pipeline.

### 1. **Pair Probability Calculation** (`pair_probability.py`)
   - Computes the pair probability between sequence elements.
   - Used for probabilistic modeling of sequence relationships.

### 2. **Future Count Calculation** (`future_count.py`)
   - Computes the future count for sequence elements.
   - Helps in predictive modeling of sequence patterns.

### 3. **Entropy Calculation** (`entropy.py`)
   - Calculates entropy for given sequence data.
   - Measures sequence variability and uncertainty.

### 4. **Sequence Similarity Calculation** (`sequence_similarity.py`)
   - Computes cosine similarity between two sequences.
   - Useful for sequence clustering and similarity-based analyses.

## Local Calculation Scripts
These scripts perform auxiliary tasks such as data combination and phylogenetic comparisons.

### 1. **Combine CSV Files** (`combine_csv.py`)
   - Merges multiple CSV files into a single dataset.
   - Useful for preprocessing large datasets.

### 2. **Combine Feature 1 Data with Feature 3** (`combine_f1_with_f3.py`)
   - Merges Feature 1 (Pair Probability) and Feature 3 (Entropy) data.
   - Saves the combined data in a new folder as a CSV file.

### 3. **Phylogenetics Comparison** (`phylogenetics_compare.py`)
   - Compares two sequences based on their phylogenetic tree structure.
   - Useful for evolutionary analysis.

### 4. **Combine Feature 1 and 2 with Phylogenetic Data** (`combine_f1f2_with_phylogenetics_data.py`)
   - Merges combined Feature 1 and Feature 2 data with phylogenetics data.
   - Saves the final dataset as a CSV file for further analysis.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scripts.git
   cd scripts
   ```
2. Navigate to the appropriate folder and run the desired script:
   ```bash
   cd "Main Calculation"
   python pair_probability.py
   ```
   or
   ```bash
   cd "Local Calculation"
   python combine_csv.py
   ```

## Contributions
Contributions are welcome! If you have additional scripts or improvements, feel free to submit a pull request.

## License
This repository is available under [MIT License](LICENSE).

## Contact
For any inquiries, please reach out via [webserver.iiserkol@gmail.com] or create an issue in this repository.
