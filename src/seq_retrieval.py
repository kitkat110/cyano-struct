#!/usr/bin/env python3

import logging
import os
import io
from Bio import Entrez, SeqIO

Entrez.email = "A.N.Other@example.com"


# -------------------------
# Functions
# -------------------------
def search_ncbi(search_term: str) -> list:
    """
    Searches the NCBI protein database for matching search terms and retrieves the record IDs.

    Args:
    search_term: A query string used to search Entrez databases.

    Returns:
        id_list: A list of the matching record IDs.
    """
    
    logging.debug(f"Search for matching NCBI protein records for {search_term}")
    with Entrez.esearch(db="protein", term=search_term) as h:
        results = Entrez.read(h)
        id_list = results["IdList"]
    if len(id_list) == 0:
        logging.warning(f"No matching records for {search_term}")

    return id_list

def fetch_sequences(id_list: list) -> list:
    """
    Retrieves the sequences of matching IDs and returns them in a FASTA format.

    Args:
    id_list: A list of the matching record IDs.

    Returns:
        seq_list: A list of sequences of matching IDs in FASTA format.
    """

    with Entrez.efetch(db="protein", id=id_list, rettype="fasta", retmode="text", rtmax=50) as h:
        record = SeqIO.parse(h, "fasta")
        seq_list = list(record)
        return seq_list
  
def save_sequences(sequences: list, output_file: str) -> None:
    """
    Writes the list of sequences to a output FASTA file.

    Args:
    sequences: A list of sequences of matching IDs in FASTA format.
    output_file: Name of the output FASTA file.

    Returns:
        None: This function does not return a value; it writes output to disk.
    """

    with open(output_file, "w") as f:
        SeqIO.write(sequences, f, "fasta") 
