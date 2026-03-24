#!/usr/bin/env python3

import logging
import os
import io
from Bio.SeqIO.FastaIO import SimpleFastaParser
import numpy as np
import pandas as pd

sequences = []

# -------------------------
# Functions
# -------------------------
def calc_metrics(input_file: str) -> None:
    """
    Takes in the aligned sequences FASTA file and parses the sequences to calculate conservation score and shannon entropy

    Args:
        input_file: Name of aligned sequences FASTA file

    Returns:
        None: This function does not return a value; it writes output to disk.
    """

    with open(input_file, "r") as f:
        for header, sequence in SimpleFastaParser(f):
            sequences.append(sequence)
    
    if len(sequences) == 0:
        logging.error(f"Unable to retrieve sequences from {input_file}")
        sys.exit(1)

    n_position = len(sequences[0])
    results = []

    for pos in range(n_position):
        residues = [seq[pos] for seq in sequences if seq[pos] != '-']
    
        if not residues:
            conservation_score = 0
            shannon_entropy = 0
        else:
            counts = {}
            for res in residues:
                counts[res] = counts.get(res, 0) + 1
        
            total = len(residues)

            # Calculate conservation score - indicates evolutionary preservation of an amino acid across samples
            max_count = max(counts.values())
            conservation_score = max_count / total # High score = high conservation (important functional sites), low score = high variability

            # Calculates Shannon entropy - measure of "uncertainty" at a single column within the alignment
            shannon_entropy = 0
            for count in counts.values():
                freq = count / total
                if freq > 0:
                    shannon_entropy -= freq * np.log2(freq) # Low entropy = high conservation, high entropy = column is variable

        results.append({
            "Position": pos + 1,
            "Conservation Score": conservation_score,
            "Shannon Entropy": shannon_entropy
        })

    pd.DataFrame(results).to_csv("data/seq_calc_results.csv")
