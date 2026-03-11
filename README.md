# CyanoStruct 🚧

The directory contains all the files for **CyanoStruct**, an interactive dashboard for predicting whether an amino acid mutation in a toxin biosynthesis protein is likely to be disruptive or tolerated.

## About
Cyanobacterial harmful algal blooms are a growing ecological and public health concern in freshwater creeks. Some species, such as Microcystis aeruginosa, produce the hepatotoxin microcystin through enzymes encoded by the mcy gene cluster. Mutations in these toxin biosynthesis genes may influence enzyme structure and function, potentially affecting toxin production. However, experimentally evaluating the impact of every possible mutation is costly and time-consuming.

CyanoStruct integrates evolutionary sequence analysis and machine learning to estimate mutation impact. Protein sequences for the mcyA gene were retrieved from the NCBI database using the Entrez API. These sequences were then aligned using the multiple alignment program MAFFT to identify conserved and variable regions. 

From the alignment, evolutionary features such as conservation scores and Shannon entropy will be be calculated for each residue position. Additional biochemical features will be computed, inclusing BLOSUM substitution scores and changes in amino acid class. These features will be used as input to a Random Forest classifier model to predict mutation impact.

## Directory Structure
```
my-mbs337-repo/
└── cyano-struct/
    ├── data/
    │   ├── aligned_microcystis_sequences.fasta
    │   └── microcystis_sequences.fasta
    ├── src/
    │   ├── feature_extraction.py
    │   ├── mut_impact_pred.py
    │   └── seq_alignment.py
    ├── Dockerfile
    └── README.md
```

## Installation
1. Clone repository

2. Build Docker container
``` bash
docker build -t cyano-struct .
```

3. Create local data folder
``` bash
mkdir data
```

## Usage
Run full pipeline in Docker
``` bash
docker run -v $(pwd)/data:/code/data cyano-struct
```
- `-v $(pwd)/data:/code/data` mounts a local folder so you can see output files
- Inside Docker, all scripts write to `/code/data` by default

## Citation/Reference
- Microcystis aeruginosa genome and toxin genes: NCBI Protein Database
- MAFFT alignment tool: https://mafft.cbrc.jp/alignment/software/  
- Biopython: https://biopython.org/