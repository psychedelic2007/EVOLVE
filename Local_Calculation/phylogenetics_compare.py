from Bio import SeqIO

def compare_sequences(seq1, seq2):
    """
    Compare two sequences and return a list of 0s and 1s indicating differences.
    """
    comparison = []
    for residue1, residue2 in zip(seq1, seq2):
        if residue1 == residue2:
            comparison.append(0)
        else:
            comparison.append(1)
    return comparison

def write_comparison_to_csv(comparison, output_file):
    """
    Write the comparison results to a CSV file.
    """
    with open(output_file, 'w') as f:
        f.write("Position,Comparison\n")
        for position, result in enumerate(comparison, start=1):
            f.write(f"{position},{result}\n")

def extract_sequence_from_fasta(fasta_file, accession_id):
    """
    Extract a sequence from a FASTA file given its accession ID.
    """
    for record in SeqIO.parse(fasta_file, "fasta"):
        if record.id == accession_id:
            return str(record.seq)
    return None

# Main FASTA file containing all sequences
main_fasta_file = "<name_of_fasta_file_containing_sequences.fasta>"

# Accession IDs of the sequences to be compared
accession_id1 = "UXX33696.1" #example. This should correspond to accession ID of sequence which you are comparing
accession_id2 = "UAJ82243.1" #example. This should corerspond to accession ID of sequence with which you are comparing

# Extract sequences from the main FASTA file
sequence1 = extract_sequence_from_fasta(main_fasta_file, accession_id1)
sequence2 = extract_sequence_from_fasta(main_fasta_file, accession_id2)

if sequence1 is None:
    print(f"Sequence with Accession ID '{accession_id1}' not found in the main FASTA file.")
elif sequence2 is None:
    print(f"Sequence with Accession ID '{accession_id2}' not found in the main FASTA file.")
else:
    # Compare the sequences
    comparison = compare_sequences(sequence1, sequence2)

    # Write the comparison results to a CSV file
    output_csv = f"{accession_id1}_comparison.csv"
    write_comparison_to_csv(comparison, output_csv)
    print(f"Comparison results have been written to '{output_csv}'.")
