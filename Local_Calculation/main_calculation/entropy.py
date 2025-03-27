import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
from Bio import AlignIO, Align
from scipy.stats import entropy

class ShannonEntropyCalculator:
    def __init__(self):
        """
        Initialize the Shannon Entropy Calculator.
        """
        pass

    def calculate_shannon_entropy(self, list_input: List[str]) -> float:
        """
        Calculate Shannon Entropy for a given list of amino acids.
        
        Args:
            list_input (List[str]): List of amino acids in a column
        
        Returns:
            float: Shannon Entropy value
        """
        unique_aa, counts = np.unique(list_input, return_counts=True)
        probabilities = counts / len(list_input)
        shannon_entropy = entropy(probabilities, base=2)
        return shannon_entropy

    def calculate_entropy_profile(self, alignment: Align.MultipleSeqAlignment) -> List[float]:
        """
        Calculate Shannon Entropy for each column in the alignment.
        
        Args:
            alignment (Align.MultipleSeqAlignment): Multiple Sequence Alignment
        
        Returns:
            List[float]: List of Shannon Entropy values for each column
        """
        shannon_entropy_list = [
            self.calculate_shannon_entropy(list(alignment[:, col_no]))
            for col_no in range(len(alignment[0]))
        ]
        
        return shannon_entropy_list

    def plot_entropy_profile(self, 
                              entropy_values: List[float], 
                              output_file: str = None, 
                              title: str = "Shannon's Entropy Profile") -> None:
        """
        Plot Shannon Entropy profile.
        
        Args:
            entropy_values (List[float]): List of entropy values
            output_file (str, optional): Path to save the plot
            title (str, optional): Plot title
        """
        plt.figure(figsize=(18, 10))
        plt.plot(entropy_values, 'r')
        plt.title(title, fontsize=16)
        plt.xlabel('Residue Position', fontsize=16)
        plt.ylabel("Shannon's Entropy", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def export_entropy_data(self, 
                             entropy_values: List[float], 
                             output_file: str = 'shannon_entropy.csv') -> None:
        """
        Export entropy values to a CSV file.
        
        Args:
            entropy_values (List[float]): List of entropy values
            output_file (str, optional): Path to save the CSV
        """
        df = pd.DataFrame({
            'Residue_Position': range(1, len(entropy_values) + 1),
            'Shannon_Entropy': entropy_values
        })
        
        df.to_csv(output_file, index=False)
        print(f"Entropy data exported to {output_file}")

def main():
    """
    Main function to run Shannon Entropy calculation and visualization.
    """
    while True:
        input_file = input("Enter the path to your Multiple Sequence Alignment file (clustal/fasta): ").strip()
    
        if os.path.exists(input_file) and input_file.lower().endswith(('.aln', '.clustal', '.fasta', '.fa')):
            break
        else:
            print("Invalid file. Please enter a valid path to a clustal or fasta alignment file.")
    
    file_format = 'clustal' if input_file.lower().endswith(('.aln', '.clustal')) else 'fasta'
    
    output_dir = input("Enter output directory (press Enter for current directory): ").strip()
    output_dir = output_dir if output_dir else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        alignment = AlignIO.read(input_file, file_format)
        calculator = ShannonEntropyCalculator()
        entropy_values = calculator.calculate_entropy_profile(alignment)
        
        plot_path = os.path.join(output_dir, 'shannon_entropy_plot.png')
        calculator.plot_entropy_profile(entropy_values, output_file=plot_path)

        csv_path = os.path.join(output_dir, 'shannon_entropy_data.csv')
        calculator.export_entropy_data(entropy_values, output_file=csv_path)
        
        print("\nProcessing complete.")
        print(f"Entropy plot saved to: {plot_path}")
        print(f"Entropy data saved to: {csv_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
