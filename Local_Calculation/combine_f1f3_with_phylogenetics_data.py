import os
import pandas as pd

# Path to the directory containing comparison files
comparison_dir = "/path/to/phylogenetics/directory"

# Path to the directory containing accession files
accession_dir = "/path/to/combined/folder"

# Iterate through comparison files
for comparison_file in os.listdir(comparison_dir):
    if comparison_file.endswith("_comparison.csv"):
        accession_id = comparison_file.split("_")[0]

        # Read comparison file
        comparison_df = pd.read_csv(os.path.join(comparison_dir, comparison_file))

        # Extract the second column as "Target"
        target_column = comparison_df.iloc[:, 1]

        # Read accession file
        accession_file = os.path.join(accession_dir, f"{accession_id}.csv")
        if os.path.exists(accession_file):
            accession_df = pd.read_csv(accession_file)

            # Insert "Target" column into the accession file
            accession_df["Target"] = target_column

            # Save the updated accession file
            accession_df.to_csv(accession_file, index=False)
            print(f"Updated {accession_file} with Target column.")
        else:
            print(f"Accession file {accession_id}.csv not found.")

print("Process completed.")
