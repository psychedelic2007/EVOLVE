import os
import pandas as pd
from tqdm import tqdm

feature1_folder = r"C:\Users\satya\Downloads\feature1"
feature3_folder = r"C:\Users\satya\Downloads\feature3"
output_folder   = r"C:\Users\satya\Downloads\output"

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(feature1_folder) if f.endswith(".csv")]

for file in tqdm(files, desc="Combining CSV files"):
    path1 = os.path.join(feature1_folder, file)
    path3 = os.path.join(feature3_folder, file)

    # Skip if matching file does not exist
    if not os.path.exists(path3):
        print(f"Skipping {file}: not found in feature3 folder")
        continue

    df1 = pd.read_csv(path1)
    df3 = pd.read_csv(path3)

    # Hard validation (no silent corruption)
    required_f1 = {"Residue", "F1"}
    required_f3 = {"Residue", "F3"}

    if not required_f1.issubset(df1.columns):
        print(f"Skipping {file}: missing {required_f1 - set(df1.columns)} in feature1")
        continue

    if not required_f3.issubset(df3.columns):
        print(f"Skipping {file}: missing {required_f3 - set(df3.columns)} in feature3")
        continue

    # Merge strictly on Residue to avoid row misalignment
    merged = pd.merge(
        df1[["Residue", "F1"]],
        df3[["Residue", "F3"]],
        on="Residue",
        how="inner"
    )

    # Rename F3 → F2 as requested
    merged.rename(columns={"F3": "F2"}, inplace=True)

    output_file = os.path.join(output_folder, file)
    merged.to_csv(output_file, index=False)
