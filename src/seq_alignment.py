import io
import subprocess
from Bio import Entrez, SeqIO

Entrez.email = "A.N.Other@example.com"

OUTPUT_FILE = "data/microcystis_sequences.fasta"
ALIGNED_FILE = "data/aligned_microcystis_sequences.fasta"


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
    cmd = ["mafft", "--auto", input_fasta]

    with open(output_fasta, "w") as outfile:
        subprocess.run(cmd, stdout=outfile, check=True)

def main():
    ids = search_ncbi()
    seqs = fetch_sequences(ids)
    save_sequences(seqs, OUTPUT_FILE)
    mafft_align_fasta(OUTPUT_FILE, ALIGNED_FILE)

if __name__ == "__main__":
    main()