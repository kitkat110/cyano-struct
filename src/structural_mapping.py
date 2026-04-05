#!/usr/bin/env python3

import pandas as pd
from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.Align import PairwiseAligner

pdb_list = PDBList()

pdb_list.retrieve_pdb_file("8JBR", file_format="mmCif", pdir="data/")

parser = MMCIFParser()

with open("data/8jbr.cif", "r") as f:
    structure = parser.get_structure("8jbr", f)

# Convert 3-letter to 1-letter amino acid codes
aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

pdb_sequence = ""
residue_nums = []
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] == ' ': # Standard amino acid
                resname = residue.get_resname()
                if resname in aa_codes:
                    pdb_sequence += aa_codes[resname]
                    residue_nums.append(residue.get_id()[1])
        break
    break


sequences = []
with open("data/aligned_microcystis_sequences.fasta", "r") as f:
        for header, sequence in SimpleFastaParser(f):
            sequences.append(sequence)

aligner = PairwiseAligner()
aligner.mode = 'local'

# Scoring
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -0.5
aligner.extend_gap_score = -0.1

for seq in sequences:
    full_seq = str(seq).replace('-','')

    alignments = aligner.align(full_seq, pdb_sequence)

    if not alignments:
        continue
    
    best = alignments[0]

    if best.score < 50:
        continue
    
    aligned_blocks = best.aligned
    seq_blocks = aligned_blocks[0]  # Ungapped sequence
    pdb_blocks = aligned_blocks[1]

    # Get overall start/end in ungapped sequence
    start = seq_blocks[0][0]
    end = seq_blocks[-1][1]

    alignment_positions = []
    ungapped_pos = 0

    for i, residue in enumerate(str(seq)):
        if residue != '-':
            if ungapped_pos >= start and ungapped_pos < end:
                alignment_positions.append(i+1)
            ungapped_pos += 1

    break

seq_calc_results = pd.read_csv("data/seq_calc_results.csv")

results = []
for i, (pdb_resnum, align_pos) in enumerate(zip(residue_nums, alignment_positions)):
    cons_row = seq_calc_results[seq_calc_results['Position'] == align_pos]
    if not cons_row.empty:
        results.append({
            'PDB_ResNum': pdb_resnum,
            'Residue': pdb_sequence[i],
            'Conservation_Score': cons_row['Conservation Score'].iloc[0],
            'Shannon_Entropy': cons_row['Shannon Entropy'].iloc[0]
        })

mapping_df = pd.DataFrame(results)
mapping_df.to_csv("data/mapped_scores.csv", index=False)