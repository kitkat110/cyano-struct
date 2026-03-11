#!/usr/bin/env python3

import logging
import os
import io
import subprocess
from Bio import Entrez, SeqIO


# -------------------------
# Functions
# -------------------------
def mafft_align_fasta(input_fasta, output_fasta):
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    cmd = ["mafft", "--auto", "--quiet", input_fasta]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    with open(output_fasta, "w") as f:
        f.write(result.stdout)

    print(f"Alignment written to {output_fasta}")
