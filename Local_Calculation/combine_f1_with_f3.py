import os
import pandas as pd

# Define the paths to the input folders
feature1_folder = "/path/to/feature1/folder/"
feature2_folder = "/path/to/feature2/folder"

output_folder = "/path/to/combined_f1f2/folder"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in feature1 folder
for file in os.listdir(feature1_folder):
    if file.endswith(".csv"):
        # Read the CSV file from feature1 folder
        df1 = pd.read_csv(os.path.join(feature1_folder, file))
        
        # Read the corresponding CSV file from feature3 folder
        df2 = pd.read_csv(os.path.join(feature2_folder, file))
        
        # Concatenate the second column from feature3 CSV with feature1 CSV
        df1['F2'] = df2['F2']
        
        # Write the concatenated DataFrame to the output folder
        output_file = os.path.join(output_folder, file)
        df1.to_csv(output_file, index=False)
