import os
import pandas as pd

def merge_csv_files(folder1, folder2, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(folder1) if f.endswith('.csv')]
    
    for file in files:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)
        
        if os.path.exists(file_path2):
            df1 = pd.read_csv(file_path1)
            df2 = pd.read_csv(file_path2)
            merged_df = pd.merge(df1, df2, on='Position', suffixes=('_F1', '_F2'))
            merged_df.columns = ['Position', 'F1', 'F2']
            output_file_path = os.path.join(output_folder, file)
            merged_df.to_csv(output_file_path, index=False)
        else:
            print(f"File {file} not found in {folder2}")

if __name__ == "__main__":
    folder1 = input("Enter the path of the first folder: ")
    folder2 = input("Enter the path of the second folder: ")
    output_folder = input("Enter the path of the output folder: ")
    
    merge_csv_files(folder1, folder2, output_folder)
    print("CSV files have been merged and saved to the output folder.")
