from Bio import Phylo, SeqIO
import pandas as pd
import os
from tqdm import tqdm

# ======================
# INPUT FILES
# ======================
TREE_FILE = "tree.nwk"
FASTA_FILE = "aligned.fasta"
OUTPUT_DIR = "mutation_csvs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD SEQUENCES
# ======================
print("Loading sequences...")
fasta_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(FASTA_FILE, "fasta")}

lengths = {len(s) for s in fasta_seqs.values()}
if len(lengths) != 1:
    raise ValueError("Sequences are not aligned")
L = lengths.pop()

# ======================
# LOAD TREE & BUILD MAPPING
# ======================
print("Loading tree and building mapping...")
tree = Phylo.read(TREE_FILE, "newick")

# Extract core accession IDs (EPI_ISL_XXXXXXX)
def extract_accession(name):
    """Extract EPI_ISL_XXXXXXXX from any format"""
    parts = name.replace('|', '_').split('_')
    for i, part in enumerate(parts):
        if part == 'EPI' and i+2 < len(parts) and parts[i+1] == 'ISL':
            return f"EPI_ISL_{parts[i+2]}"
    return None

# Build mapping: tree_name -> fasta_id
tree_to_fasta = {}
for leaf in tree.get_terminals():
    tree_name = leaf.name
    tree_acc = extract_accession(tree_name)
    
    if tree_acc:
        # Find matching FASTA ID
        for fasta_id in fasta_seqs.keys():
            if tree_acc in fasta_id:
                tree_to_fasta[tree_name] = fasta_id
                break

print(f"Mapped {len(tree_to_fasta)}/{len(list(tree.get_terminals()))} tree leaves to FASTA")

# Verify mapping
print("\nMapping sample:")
for i, (tree_name, fasta_id) in enumerate(list(tree_to_fasta.items())[:3]):
    print(f"  {tree_name}")
    print(f"  -> {fasta_id}\n")

# ======================
# COMPARISON FUNCTION
# ======================
def compare_and_save(tree_query, tree_ref, pbar):
    if tree_query not in tree_to_fasta or tree_ref not in tree_to_fasta:
        pbar.write(f"SKIP: {tree_query} or {tree_ref} not mapped")
        return
    
    fasta_query = tree_to_fasta[tree_query]
    fasta_ref = tree_to_fasta[tree_ref]
    
    s1 = fasta_seqs[fasta_query]
    s2 = fasta_seqs[fasta_ref]
    
    df = pd.DataFrame({
        "Residue": range(1, L + 1),
        "target": [0 if s1[i] == s2[i] else 1 for i in range(L)]
    })
    
    # Use tree name for output filename (clean it up)
    clean_name = tree_query.split('_')[0]  # Just use the number prefix
    out = os.path.join(OUTPUT_DIR, f"{clean_name}.csv")
    df.to_csv(out, index=False)
    pbar.write(f"✓ {clean_name}: {tree_query} vs {tree_ref}")

# ======================
# TRAVERSAL
# ======================
def get_all_leaves(clade):
    return [t.name for t in clade.get_terminals() if t.name in tree_to_fasta]

# First pass: count total comparisons
print("\nCounting total comparisons...")
processed_comparisons = set()
comparison_list = []

for clade in tree.find_clades(order="postorder"):
    if clade.is_terminal():
        continue
    
    if len(clade.clades) != 2:
        continue
    
    left_clade, right_clade = clade.clades
    left_leaves = get_all_leaves(left_clade)
    right_leaves = get_all_leaves(right_clade)
    
    if not left_leaves or not right_leaves:
        continue
    
    reference = right_leaves[0]
    
    for query in left_leaves:
        pair = tuple(sorted([query, reference]))
        if pair not in processed_comparisons:
            comparison_list.append((query, reference))
            processed_comparisons.add(pair)

# Second pass: execute with progress bar
print(f"\nExecuting {len(comparison_list)} comparisons...")
with tqdm(total=len(comparison_list), desc="Comparing sequences", unit="pair") as pbar:
    for query, reference in comparison_list:
        compare_and_save(query, reference, pbar)
        pbar.update(1)

print(f"\n✓ Complete! Total comparisons: {len(comparison_list)}")
print(f"Output directory: {OUTPUT_DIR}")
