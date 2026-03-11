#!/usr/bin/env python3

import logging
import os
import io
import subprocess
from Bio import Entrez, SeqIO


# -------------------------
# Functions
# -------------------------
def mafft_align_fasta(input_fasta: str, output_fasta: str) -> None:
    """
    Aligns the sequences in the input FASTA file and writes the results to an output FASTA file.

    Args:
        input_fasta: Name of the input FASTA file.
        output_fasta: Name of the output FASTA file.

    Returns:
        None: This function does not return a value; it writes output to disk.
    """

    try:
        cmd = ["mafft", "--auto", "--quiet", input_fasta]
    except FileNotFoundError:
        logging.error(f"Input FASTA file {input_fasta} not found. Exiting.")
        sys.exit(1)

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    with open(output_fasta, "w") as f:
        f.write(result.stdout)

    print(f"Alignment written to {output_fasta}")
