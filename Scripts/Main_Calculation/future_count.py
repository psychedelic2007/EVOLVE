import os
import csv
import zipfile
from typing import List, Dict
from tqdm import tqdm
from Bio import SeqIO

# =========================
# USER CONFIG (EDIT THESE)
# =========================
FASTA_FILE = "cleaned_sequences.fasta"
OUTPUT_DIR = "feature3"
ZIP_NAME = "feature3.zip"
# =========================


class Feature3Calculator:
    def __init__(self):
        self.aa_list = [
            'A','C','D','E','F','G','H','I','K','L',
            'M','N','P','Q','R','S','T','V','W','Y'
        ]

        self.future_value_coeffs = {
            'A': [(12,'A'),(2,'D'),(2,'E'),(4,'G'),(4,'P'),(4,'S'),(4,'T'),(4,'V')],
            'C': [(2,'R'),(2,'C'),(2,'G'),(2,'F'),(4,'S'),(2,'W'),(2,'Y')],
            'D': [(2,'A'),(2,'N'),(2,'D'),(4,'E'),(2,'G'),(2,'H'),(2,'Y'),(2,'V')],
            'E': [(2,'A'),(4,'D'),(2,'E'),(2,'Q'),(2,'G'),(2,'K'),(2,'V')],
            'F': [(2,'C'),(2,'I'),(6,'L'),(2,'F'),(2,'S'),(2,'Y'),(2,'V')],
            'G': [(4,'A'),(6,'R'),(2,'D'),(2,'C'),(2,'E'),(12,'G'),(2,'S'),(1,'W'),(4,'V')],
            'H': [(2,'R'),(2,'N'),(2,'D'),(4,'Q'),(2,'H'),(2,'L'),(2,'P'),(2,'Y')],
            'I': [(1,'R'),(2,'N'),(6,'I'),(4,'L'),(1,'K'),(3,'M'),(2,'F'),(2,'S'),(3,'T'),(3,'V')],
            'K': [(2,'R'),(4,'N'),(2,'E'),(2,'Q'),(1,'I'),(2,'K'),(1,'M'),(2,'T')],
            'L': [(4,'R'),(2,'Q'),(2,'H'),(4,'I'),(18,'L'),(2,'M'),(6,'F'),(4,'P'),(2,'S'),(1,'W'),(6,'V')],
            'M': [(1,'R'),(3,'I'),(2,'L'),(1,'K'),(1,'T'),(1,'V')],
            'N': [(2,'N'),(2,'D'),(2,'H'),(2,'I'),(4,'K'),(2,'S'),(2,'T'),(2,'Y')],
            'P': [(4,'A'),(4,'D'),(2,'Q'),(2,'H'),(4,'L'),(12,'P'),(4,'S'),(4,'T')],
            'Q': [(2,'R'),(2,'E'),(2,'Q'),(4,'H'),(2,'L'),(2,'K'),(2,'P')],
            'R': [(18,'R'),(2,'C'),(2,'Q'),(6,'G'),(2,'H'),(1,'I'),(4,'L'),(2,'K'),
                  (1,'M'),(4,'P'),(6,'S'),(2,'T'),(2,'W')],
            'S': [(4,'A'),(6,'R'),(2,'N'),(4,'C'),(2,'G'),(2,'I'),(2,'L'),(2,'F'),
                  (4,'P'),(14,'S'),(6,'T'),(1,'W'),(2,'Y')],
            'T': [(4,'A'),(2,'R'),(2,'N'),(3,'I'),(2,'K'),(1,'M'),(4,'P'),(6,'S'),(12,'T')],
            'V': [(4,'A'),(2,'D'),(2,'E'),(4,'G'),(3,'I'),(6,'L'),(1,'M'),(2,'F'),(12,'V')],
            'W': [(2,'R'),(2,'C'),(1,'G'),(1,'L'),(1,'S')],
            'Y': [(2,'N'),(2,'D'),(2,'C'),(2,'H'),(2,'F'),(2,'S'),(2,'Y')]
        }

    def calculate_future_values(self, aa_counts: Dict[str, int], seq_len: int) -> List[float]:
        values = []
        for aa in self.aa_list:
            w, d = 0, 0
            for coeff, rel in self.future_value_coeffs[aa]:
                w += coeff * aa_counts.get(rel, 0)
                d += coeff
            values.append(w / (d * seq_len) if seq_len > 0 else 0)
        return values

    def run(self, fasta_file: str, output_dir: str, zip_name: str):
        os.makedirs(output_dir, exist_ok=True)
        zip_path = os.path.join(output_dir, zip_name)

        records = list(SeqIO.parse(fasta_file, "fasta"))
        if not records:
            raise ValueError("FASTA file is empty.")

        with zipfile.ZipFile(zip_path, "w") as z:
            for rec in tqdm(records, desc="Processing sequences"):
                seq = str(rec.seq)
                aa_counts = {aa: seq.count(aa) for aa in self.aa_list}

                curr = [aa_counts[a] / len(seq) if len(seq) > 0 else 0 for a in self.aa_list]
                fut = self.calculate_future_values(aa_counts, len(seq))
                fcaa = [(f/c if c > 0 else 0) for f, c in zip(fut, curr)]

                accession = rec.id.split("|")[0]
                accession = ''.join(c for c in accession if c.isalnum() or c in "_-")
                csv_name = f"{accession}.csv"
                csv_path = os.path.join(output_dir, csv_name)

                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Residue", "F3"])
                    for r in seq:
                        w.writerow([r, fcaa[self.aa_list.index(r)]])

                z.write(csv_path, arcname=csv_name)
                os.remove(csv_path)

        return zip_path


# =========================
# EXECUTION (RUN THIS CELL)
# =========================
calculator = Feature3Calculator()
zip_output = calculator.run(FASTA_FILE, OUTPUT_DIR, ZIP_NAME)

print(f"✔ Feature 3 completed. ZIP saved at:\n{zip_output}")
