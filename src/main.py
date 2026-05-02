#!/usr/bin/env python3

import os
import logging
import argparse
import socket

from seq_retrieval import search_ncbi, fetch_sequences, save_sequences
from seq_alignment import mafft_align_fasta
from seq_calcs import calc_metrics
from structural_mapping import (
    download_pdb,
    parse_pdb_sequence,
    get_alignment_positions,
    map_conservation_scores
)
from train_gmm import (
    build_mutation_dataset,
    train_gmm,
    plot_cluster_heatmap,
    save_models
)

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
# Constants
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEARCH_TERM = '"Microcystis"[Organism] AND mcyA AND 700:3000[SLEN]'
OUTPUT_FILE = os.path.join(DATA_DIR, "microcystis_sequences.fasta")
ALIGNED_FILE = os.path.join(DATA_DIR, "aligned_microcystis_sequences.fasta")
CIF_FILE = os.path.join(DATA_DIR, "8jbr.cif")
SEQ_CALC_FILE = os.path.join(DATA_DIR, "seq_calc_results.csv")
MAPPED_FILE = os.path.join(DATA_DIR, "mapped_scores.csv")

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

    logging.info("Calculating sequence metrics")
    calc_metrics(ALIGNED_FILE)

    logging.info("Downloading PDB structure")
    download_pdb(pdb_id='8JBR', output_dir=DATA_DIR)

    logging.info("Parsing PDB sequence")
    pdb_sequence, residue_nums = parse_pdb_sequence(cif_path=CIF_FILE)

    logging.info("Aligning sequences to PDB structure")
    alignment_positions = get_alignment_positions(fasta_path=ALIGNED_FILE, pdb_sequence=pdb_sequence)

    logging.info("Mapping conservation scores to PDB residues")
    map_conservation_scores(pdb_sequence=pdb_sequence, residue_nums=residue_nums, alignment_positions=alignment_positions,
                            seq_calc_path=SEQ_CALC_FILE, output_path=MAPPED_FILE)

    logging.info("Building mutation dataset")
    mut_df = build_mutation_dataset(mapped_scores_path=MAPPED_FILE)

    logging.info("Training GMM")
    gmm, scaler, cluster_names, mut_df = train_gmm(mut_df)

    logging.info("Plotting cluster heatmap")
    plot_cluster_heatmap(mut_df)

    logging.info("Saving models")
    save_models(gmm, scaler, cluster_names)

    logging.info("Pipeline complete — run app.py to launch the dashboard")

if __name__ == "__main__":
    main()