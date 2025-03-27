import os
import pandas as pd

# input folders
feature1_folder = "/path/to/feature1/folder/"
feature2_folder = "/path/to/feature2/folder"

output_folder = "/path/to/combined_f1f2/folder"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file in os.listdir(feature1_folder):
    if file.endswith(".csv"):
        df1 = pd.read_csv(os.path.join(feature1_folder, file))
        df2 = pd.read_csv(os.path.join(feature2_folder, file))
        df1['F2'] = df2['F2']
        output_file = os.path.join(output_folder, file)
        df1.to_csv(output_file, index=False)
