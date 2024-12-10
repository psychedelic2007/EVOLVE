import os
import pandas as pd

comparison_dir = "/path/to/phylogenetics/directory"
accession_dir = "/path/to/combined/folder"

for comparison_file in os.listdir(comparison_dir):
    if comparison_file.endswith("_comparison.csv"):
        accession_id = comparison_file.split("_")[0]

        comparison_df = pd.read_csv(os.path.join(comparison_dir, comparison_file))
        target_column = comparison_df.iloc[:, 1]

        accession_file = os.path.join(accession_dir, f"{accession_id}.csv")
        if os.path.exists(accession_file):
            accession_df = pd.read_csv(accession_file)
            accession_df["Target"] = target_column
            accession_df.to_csv(accession_file, index=False)
            print(f"Updated {accession_file} with Target column.")
        else:
            print(f"Accession file {accession_id}.csv not found.")

print("Process completed.")
