#!/usr/bin/env python3

import logging
import os
import io
from Bio.SeqIO.FastaIO import SimpleFastaParser
import numpy as np
import pandas as pd

sequences = []

with open("data/aligned_microcystis_sequences.fasta", "r") as f:
    for header, sequence in SimpleFastaParser(f):
        sequences.append(sequence)

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

        max_count = max(counts.values())
        conservation_score = max_count / total

        shannon_entropy = 0
        for count in counts.values():
            freq = count / total
            if freq > 0:
                shannon_entropy -= freq * np.log2(freq)

    results.append({
        "Position": pos + 1,
        "Conservation Score": conservation_score,
        "Shannon Entropy": shannon_entropy
    })

pd.DataFrame(results).to_csv("data/seq_calc_results.csv")
