from Bio import Entrez, SeqIO

Entrez.email = "A.N.Other@example.com"

ORGANISM = "Microcystis aeruginosa"
GENE = "mcyA"
search_term = f"{ORGANISM}[Organism] AND {GENE}[Gene]"

with Entrez.esearch(db="protein", term=search_term) as h:
    results = Entrez.read(h)
    id_list = results["IdList"]

gb_rec = None
with Entrez.efetch(db="protein", id=id_list, rettype="gb", retmode="text", rtmax=10) as h:
    record = SeqIO.parse(h, "gb")
    rec_list = list(record)
    gb_rec = rec_list[0]
    print(f"ID: {gb_rec.id}\nName: {gb_rec.name}\nDescription: {gb_rec.descriptiion}\nSequence: {gb_rec.seq}")

