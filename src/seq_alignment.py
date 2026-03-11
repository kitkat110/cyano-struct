#!/usr/bin/env python3

import os
import io
import subprocess
from Bio import Entrez, SeqIO

Entrez.email = "A.N.Other@example.com"

DATA_DIR = os.path.join(os.getcwd(), "data")  # always /code/data inside container
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "microcystis_sequences.fasta")
ALIGNED_FILE = os.path.join(DATA_DIR, "aligned_microcystis_sequences.fasta")


def search_ncbi():
    search_term = '"Microcystis aeruginosa"[Organism] AND mcyA AND 400:2000[SLEN]'
    
    with Entrez.esearch(db="protein", term=search_term) as h:
        results = Entrez.read(h)
        id_list = results["IdList"]
        return id_list


def fetch_sequences(id_list):
    gb_rec = None
    with Entrez.efetch(db="protein", id=id_list, rettype="fasta", retmode="text", rtmax=40) as h:
        record = SeqIO.parse(h, "fasta")
        rec_list = list(record)
        return rec_list

        
def save_sequences(sequences, output_file):
    with open(output_file, "w") as f:
        SeqIO.write(sequences, f, "fasta") 

def mafft_align_fasta(input_fasta, output_fasta):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    # Build the command
    cmd = ["mafft", "--auto", "--quiet", input_fasta]

    # Run MAFFT and capture stdout
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Write stdout (the alignment) to output_fasta
    with open(output_fasta, "w") as f:
        f.write(result.stdout)

    print(f"Alignment written to {output_fasta}")

def main():
    ids = search_ncbi()
    seqs = fetch_sequences(ids)
    save_sequences(seqs, OUTPUT_FILE)
    mafft_align_fasta(OUTPUT_FILE, ALIGNED_FILE)

if __name__ == "__main__":
    main()