import os
import pandas as pd

def merge_csv_files(folder1, folder2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all CSV files in the first folder
    files = [f for f in os.listdir(folder1) if f.endswith('.csv')]
    
    for file in files:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)
        
        # Check if the corresponding file exists in the second folder
        if os.path.exists(file_path2):
            # Read the CSV files
            df1 = pd.read_csv(file_path1)
            df2 = pd.read_csv(file_path2)
            
            # Merge the dataframes on the 'Position number' column
            merged_df = pd.merge(df1, df2, on='Position', suffixes=('_F1', '_F2'))
            
            # Rename columns appropriately
            merged_df.columns = ['Position', 'F1', 'F2']
            
            # Save the merged dataframe to the output folder
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
