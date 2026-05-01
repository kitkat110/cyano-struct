#!/usr/bin/env python3

import pandas as pd
from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.Align import PairwiseAligner

AA_CODES = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def download_pdb(pdb_id='8JBR', output_dir='data/'):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb_id, file_format="mmCif", pdir=output_dir)
    print(f"Downloaded {pdb_id} to {output_dir}")

def parse_pdb_sequence(cif_path='data/8jbr.cif'):
    parser = MMCIFParser(QUIET=True)
    with open(cif_path, 'r') as f:
        structure = parser.get_structure("8jbr", f)

    pdb_sequence = ""
    residue_nums = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':
                    resname = residue.get_resname()
                    if resname in AA_CODES:
                        pdb_sequence += AA_CODES[resname]
                        residue_nums.append(residue.get_id()[1])
            break
        break

    print(f"Parsed {len(pdb_sequence)} residues from PDB structure")
    return pdb_sequence, residue_nums

def get_alignment_positions(fasta_path, pdb_sequence):
    sequences = []
    with open(fasta_path, 'r') as f:
        for header, sequence in SimpleFastaParser(f):
            sequences.append(sequence)

    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    for seq in sequences:
        full_seq = str(seq).replace('-', '')
        alignments = aligner.align(full_seq, pdb_sequence)

        if not alignments:
            continue

        best = alignments[0]
        if best.score < 50:
            continue

        aligned_blocks = best.aligned
        seq_blocks = aligned_blocks[0]
        start = seq_blocks[0][0]
        end = seq_blocks[-1][1]

        alignment_positions = []
        ungapped_pos = 0
        for i, residue in enumerate(str(seq)):
            if residue != '-':
                if ungapped_pos >= start and ungapped_pos < end:
                    alignment_positions.append(i + 1)
                ungapped_pos += 1

        print(f"Found {len(alignment_positions)} aligned positions")
        return alignment_positions

    raise ValueError("No suitable alignment found in FASTA file")

def map_conservation_scores(pdb_sequence, residue_nums, alignment_positions,
                             seq_calc_path='data/seq_calc_results.csv',
                             output_path='data/mapped_scores.csv'):
    seq_calc_results = pd.read_csv(seq_calc_path)

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
    mapping_df.to_csv(output_path, index=False)
    print(f"Saved {len(mapping_df)} mapped residues to {output_path}")
    return mapping_df