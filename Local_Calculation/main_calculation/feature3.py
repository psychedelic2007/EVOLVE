import os
import csv
import re
import zipfile
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq

class Feature3Calculator:
    def __init__(self):
        """
        Initialize the Feature3Calculator with amino acid lists and future value coefficients.
        """
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # Future value coefficients for each amino acid
        self.future_value_coeffs = {
            'A': [(12, 'A'), (2, 'D'), (2, 'E'), (4, 'G'), (4, 'P'), (4, 'S'), (4, 'T'), (4, 'V')],
            'C': [(2, 'R'), (2, 'C'), (2, 'G'), (2, 'F'), (4, 'S'), (2, 'W'), (2, 'Y')],
            'D': [(2, 'A'), (2, 'N'), (2, 'D'), (4, 'E'), (2, 'G'), (2, 'H'), (2, 'Y'), (2, 'V')],
            'E': [(2, 'A'), (4, 'D'), (2, 'E'), (2, 'Q'), (2, 'G'), (2, 'K'), (2, 'V')],
            'F': [(2, 'C'), (2, 'I'), (6, 'L'), (2, 'F'), (2, 'S'), (2, 'Y'), (2, 'V')],
            'G': [(4, 'A'), (6, 'R'), (2, 'D'), (2, 'C'), (2, 'E'), (12, 'G'), (2, 'S'), (1, 'W'), (4, 'V')],
            'H': [(2, 'R'), (2, 'N'), (2, 'D'), (4, 'Q'), (2, 'H'), (2, 'L'), (2, 'P'), (2, 'Y')],
            'I': [(1, 'R'), (2, 'N'), (6, 'I'), (4, 'L'), (1, 'K'), (3, 'M'), (2, 'F'), (2, 'S'), (3, 'T'), (3, 'V')],
            'K': [(2, 'R'), (4, 'N'), (2, 'E'), (2, 'Q'), (1, 'I'), (2, 'K'), (1, 'M'), (2, 'T')],
            'L': [(4, 'R'), (2, 'Q'), (2, 'H'), (4, 'I'), (18, 'L'), (2, 'M'), (6, 'F'), (4, 'P'), (2, 'S'), (1, 'W'), (6, 'V')],
            'M': [(1, 'R'), (3, 'I'), (2, 'L'), (1, 'K'), (1, 'T'), (1, 'V')],
            'N': [(2, 'N'), (2, 'D'), (2, 'H'), (2, 'I'), (4, 'K'), (2, 'S'), (2, 'T'), (2, 'Y')],
            'P': [(4, 'A'), (4, 'D'), (2, 'Q'), (2, 'H'), (4, 'L'), (12, 'P'), (4, 'S'), (4, 'T')],
            'Q': [(2, 'R'), (2, 'E'), (2, 'Q'), (4, 'H'), (2, 'L'), (2, 'K'), (2, 'P')],
            'R': [(18, 'R'), (2, 'C'), (2, 'Q'), (6, 'G'), (2, 'H'), (1, 'I'), (4, 'L'), (2, 'K'), 
                  (1, 'M'), (4, 'P'), (6, 'S'), (2, 'T'), (2, 'W')],
            'S': [(4, 'A'), (6, 'R'), (2, 'N'), (4, 'C'), (2, 'G'), (2, 'I'), (2, 'L'), (2, 'F'), 
                  (4, 'P'), (14, 'S'), (6, 'T'), (1, 'W'), (2, 'Y')],
            'T': [(4, 'A'), (2, 'R'), (2, 'N'), (3, 'I'), (2, 'K'), (1, 'M'), (4, 'P'), (6, 'S'), (12, 'T')],
            'V': [(4, 'A'), (2, 'D'), (2, 'E'), (4, 'G'), (3, 'I'), (6, 'L'), (1, 'M'), (2, 'F'), (12, 'V')],
            'W': [(2, 'R'), (2, 'C'), (1, 'G'), (1, 'L'), (1, 'S')],
            'Y': [(2, 'N'), (2, 'D'), (2, 'C'), (2, 'H'), (2, 'F'), (2, 'S'), (2, 'Y')]
        }

    def calculate_future_values(self, aa_counts: Dict[str, int], sequence_length: int) -> List[float]:
        """
        Calculate future values for each amino acid.
        
        Args:
            aa_counts (Dict[str, int]): Amino acid counts in the sequence
            sequence_length (int): Length of the sequence
        
        Returns:
            List[float]: Future values for each amino acid
        """
        future_values = []
        
        for aa in self.aa_list:
            total_weight = 0
            total_divisor = 0
            
            for coeff, related_aa in self.future_value_coeffs[aa]:
                total_weight += coeff * aa_counts.get(related_aa, 0)
                total_divisor += coeff
            
            # Calculate future value, handling zero length sequences
            future_value = (total_weight / (total_divisor * sequence_length)) if sequence_length > 0 else 0
            future_values.append(future_value)
        
        return future_values

    def process_sequences(self, input_file: str, output_dir: str = '.') -> str:
        """
        Process sequences from a FASTA file and generate CSV files in a zip archive.
        
        Args:
            input_file (str): Path to the input FASTA file
            output_dir (str, optional): Directory to save output. Defaults to current directory.
        
        Returns:
            str: Path to the generated zip file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_zip_path = os.path.join(output_dir, 'feature3.zip')
        
        try:
            sequences = list(SeqIO.parse(input_file, "fasta"))
        except Exception as e:
            print(f"Error reading input file: {e}")
            return ''
        
        # Process sequences
        with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            for seq in tqdm(sequences, desc="Processing Sequences", unit="sequence"):
                sequence = str(seq.seq)
                aa_counts = {aa: sequence.count(aa) for aa in self.aa_list}
                aa_curr_count_list = [
                    aa_counts[aa] / len(sequence) if len(sequence) > 0 else 0 
                    for aa in self.aa_list
                ]

                aa_fut_count_list = self.calculate_future_values(aa_counts, len(sequence))
              
                fcaa = [
                    (fut / curr if curr > 0 else 0) 
                    for fut, curr in zip(aa_fut_count_list, aa_curr_count_list)
                ]
                
                sanitized_id = re.sub(r'[^\w\-_\. ]', '_', seq.id)
                csv_filename = f"{sanitized_id}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                # Write CSV file
                with open(csv_path, "w", newline="") as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(['Sequence', 'F3'])
                    for residue in sequence:
                        index = self.aa_list.index(residue)
                        feature_value = fcaa[index]
                        writer.writerow([residue, feature_value])
                
                output_zip.write(csv_path, arcname=csv_filename)
                os.remove(csv_path)
        
        return output_zip_path

def main():
    """
    Main function to run the Feature 3 calculation script.
    """
    while True:
        input_file = input("Enter the path to your FASTA file: ").strip()
        if os.path.exists(input_file):
            break
        else:
            print("File not found. Please enter a valid file path.")
    
    output_dir = input("Enter output directory (press Enter for current directory): ").strip()
    output_dir = output_dir if output_dir else '.'
    
    calculator = Feature3Calculator()
    
    try:
        output_zip = calculator.process_sequences(input_file, output_dir)
        if output_zip:
            print(f"\nProcessing complete. Output zip file created: {output_zip}")
        else:
            print("Failed to process sequences.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
