import os
import csv
import zipfile
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

class Feature1Calculator:
    def __init__(self):
        """
        Initialize the Feature1Calculator with amino acid lists.
        """
        # Predefined list of amino acids
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.aa_pair = [f'{i}{j}' for i in self.aa_list for j in self.aa_list]

    def calculate_f1_values(self, sequence: str) -> List[float]:
        """
        Calculate F1 values for each residue in the sequence.
        
        Args:
            sequence (str): Amino acid sequence
        
        Returns:
            List[float]: F1 values for each residue
        """
        f1_values = []
        for i in range(len(sequence)):
            if i == 0:  # First residue
                if i + 1 < len(sequence):
                    d = sequence[i] + sequence[i + 1]
                    x1 = sequence.count(d) - round((sequence.count(sequence[i]) * sequence.count(sequence[i + 1])) / len(sequence))
                    f1_values.append(x1)
                else:
                    f1_values.append(0)
            elif i == len(sequence) - 1:  # Last residue
                if i - 1 >= 0:
                    d = sequence[i - 1] + sequence[i]
                    x1 = sequence.count(d) - round((sequence.count(sequence[i - 1]) * sequence.count(sequence[i])) / len(sequence))
                    f1_values.append(x1)
                else:
                    f1_values.append(0)
            else:  # Middle residues
                d1 = sequence[i - 1] + sequence[i]  # Previous pair
                d2 = sequence[i] + sequence[i + 1]  # Next pair
                x1 = sequence.count(d1) - round((sequence.count(sequence[i - 1]) * sequence.count(sequence[i])) / len(sequence))
                x2 = sequence.count(d2) - round((sequence.count(sequence[i]) * sequence.count(sequence[i + 1])) / len(sequence))
                f1_values.append((x1 + x2) / 2)  # Average of both contributions
        
        return f1_values

    def process_sequences(self, input_file: str, output_dir: str = '.') -> str:
        """
        Process sequences from a FASTA file and generate CSV files in a zip archive.
        
        Args:
            input_file (str): Path to the input FASTA file
            output_dir (str, optional): Directory to save output. Defaults to current directory.
        
        Returns:
            str: Path to the generated zip file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output zip path
        output_zip_path = os.path.join(output_dir, 'feature1.zip')
        
        # Read sequences
        try:
            sequences = list(SeqIO.parse(input_file, "fasta"))
        except Exception as e:
            print(f"Error reading input file: {e}")
            return ''
        
        # Process sequences
        with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            # Use tqdm for progress tracking
            for seq in tqdm(sequences, desc="Processing Sequences", unit="sequence"):
                # Convert sequence to string
                aa = str(seq.seq)
                
                # Calculate F1 values
                f1_values = self.calculate_f1_values(aa)
                
                # Create sanitized filename for each sequence
                sanitized_seq_id = ''.join(e if e.isalnum() else '_' for e in seq.id)
                csv_filename = f"{sanitized_seq_id}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                # Write CSV file
                with open(csv_path, "w", newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['Sequence', 'F1'])
                    for i in range(len(aa)):
                        writer.writerow([aa[i], f1_values[i]])
                
                # Add CSV file to zip archive
                output_zip.write(csv_path, arcname=csv_filename)
                
                # Remove temporary CSV file
                os.remove(csv_path)
        
        return output_zip_path

def main():
    """
    Main function to run the Feature 1 calculation script.
    """
    # Prompt user for input file
    while True:
        input_file = input("Enter the path to your FASTA file: ").strip()
        
        # Check if file exists
        if os.path.exists(input_file):
            break
        else:
            print("File not found. Please enter a valid file path.")
    
    # Prompt for output directory (optional)
    output_dir = input("Enter output directory (press Enter for current directory): ").strip()
    output_dir = output_dir if output_dir else '.'
    
    # Initialize calculator
    calculator = Feature1Calculator()
    
    try:
        # Process sequences
        output_zip = calculator.process_sequences(input_file, output_dir)
        
        if output_zip:
            print(f"\nProcessing complete. Output zip file created: {output_zip}")
        else:
            print("Failed to process sequences.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
