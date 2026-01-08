import os
import csv
from typing import List
from tqdm import tqdm
from Bio import SeqIO


class Feature1Calculator:
    def __init__(self):
        self.aa_list = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]

    def calculate_f1_values(self, sequence: str) -> List[float]:
        f1_values = []
        n = len(sequence)

        for i in range(n):
            if i == 0:
                if n > 1:
                    d = sequence[0] + sequence[1]
                    x1 = sequence.count(d) - round(
                        (sequence.count(sequence[0]) *
                         sequence.count(sequence[1])) / n
                    )
                    f1_values.append(x1)
                else:
                    f1_values.append(0.0)

            elif i == n - 1:
                d = sequence[n - 2] + sequence[n - 1]
                x1 = sequence.count(d) - round(
                    (sequence.count(sequence[n - 2]) *
                     sequence.count(sequence[n - 1])) / n
                )
                f1_values.append(x1)

            else:
                d1 = sequence[i - 1] + sequence[i]
                d2 = sequence[i] + sequence[i + 1]

                x1 = sequence.count(d1) - round(
                    (sequence.count(sequence[i - 1]) *
                     sequence.count(sequence[i])) / n
                )
                x2 = sequence.count(d2) - round(
                    (sequence.count(sequence[i]) *
                     sequence.count(sequence[i + 1])) / n
                )

                f1_values.append((x1 + x2) / 2)

        return f1_values

    def process_fasta(self, fasta_file: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if not sequences:
            raise ValueError("FASTA file contains no sequences.")

        for record in tqdm(sequences, desc="Processing sequences"):
            seq_str = str(record.seq)

            f1_values = self.calculate_f1_values(seq_str)

            # ---- CORRECT GISAID ACCESSION HANDLING ----
            # Example header:
            # >EPI_ISL_16171254|A/turkey/Minnesota/22-012123-001/2022|A_/_H5N1|2022-04-19
            accession_id = record.id.split("|")[0]

            # Defensive sanitization (minimal)
            accession_id = ''.join(c for c in accession_id if c.isalnum() or c in "_-")

            csv_path = os.path.join(output_dir, f"{accession_id}.csv")

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Position", "Residue", "F1"])
                for i, (res, val) in enumerate(zip(seq_str, f1_values), start=1):
                    writer.writerow([i, res, val])


def main():
    fasta_file = input("Enter FASTA file path: ").strip()
    if not os.path.isfile(fasta_file):
        raise FileNotFoundError("FASTA file not found.")

    output_dir = input("Enter output directory: ").strip()
    if not output_dir:
        output_dir = "feature1_results"

    calculator = Feature1Calculator()
    calculator.process_fasta(fasta_file, output_dir)

    print(f"\nDone. CSV files written to: {output_dir}")


if __name__ == "__main__":
    main()
