#!/usr/bin/env python3

import os
import logging
import argparse
import socket

from seq_retrieval import search_ncbi, fetch_sequences, save_sequences
from seq_alignment import mafft_align_fasta


# -------------------------
# Logging setup
# -------------------------
log_parser = argparse.ArgumentParser()
log_parser.add_argument(
    '-l', '--loglevel',
    type=str,
    required=False,
    default='WARNING',
    help='set log level to DEBUG, INFO, WARNING, ERROR, or CRITICAL'
)
args = log_parser.parse_args()

format_str = (
    f'[%(asctime)s {socket.gethostname()}] '
    '%(filename)s:%(funcName)s:%(lineno)s - %(levelname)s: %(message)s'
)
logging.basicConfig(level=args.loglevel, format=format_str)


# -------------------------
# Constants (configuration)
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEARCH_TERM = '"Microcystis aeruginosa"[Organism] AND mcyA AND 400:3000[SLEN]'

OUTPUT_FILE = os.path.join(DATA_DIR, "microcystis_sequences.fasta")
ALIGNED_FILE = os.path.join(DATA_DIR, "aligned_microcystis_sequences.fasta")


# -------------------------
# Pipeline
# -------------------------
def main():
    logging.info("Searching NCBI")
    ids = search_ncbi(SEARCH_TERM)

    logging.info(f"Found {len(ids)} sequences")

    logging.info("Fetching sequences")
    seqs = fetch_sequences(ids)

    logging.info("Saving FASTA")
    save_sequences(seqs, OUTPUT_FILE)

    logging.info("Running MAFFT alignment")
    mafft_align_fasta(OUTPUT_FILE, ALIGNED_FILE)

    logging.info("Pipeline complete")

if __name__ == "__main__":
    main()