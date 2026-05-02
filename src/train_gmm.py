#!/usr/bin/env python3

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import substitution_matrices
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

AA_PROPS = {
    'A': {'charge': 0, 'hydrophobicity': 1.8, 'class': 'nonpolar'},
    'R': {'charge': 1, 'hydrophobicity': -4.5, 'class': 'basic'},
    'N': {'charge': 0, 'hydrophobicity': -3.5, 'class': 'polar'},
    'D': {'charge': -1, 'hydrophobicity': -3.5, 'class': 'acidic'},
    'C': {'charge': 0, 'hydrophobicity': 2.5, 'class': 'polar'},
    'Q': {'charge': 0, 'hydrophobicity': -3.5, 'class': 'polar'},
    'E': {'charge': -1, 'hydrophobicity': -3.5, 'class': 'acidic'},
    'G': {'charge': 0, 'hydrophobicity': -0.4, 'class': 'nonpolar'},
    'H': {'charge': 0, 'hydrophobicity': -3.2, 'class': 'polar'},
    'I': {'charge': 0, 'hydrophobicity': 4.5, 'class': 'nonpolar'},
    'L': {'charge': 0, 'hydrophobicity': 3.8, 'class': 'nonpolar'},
    'K': {'charge': 1, 'hydrophobicity': -3.9, 'class': 'basic'},
    'M': {'charge': 0, 'hydrophobicity': 1.9, 'class': 'nonpolar'},
    'F': {'charge': 0, 'hydrophobicity': 2.8, 'class': 'aromatic'},
    'P': {'charge': 0, 'hydrophobicity': -1.6, 'class': 'nonpolar'},
    'S': {'charge': 0, 'hydrophobicity': -0.8, 'class': 'polar'},
    'T': {'charge': 0, 'hydrophobicity': -0.7, 'class': 'polar'},
    'W': {'charge': 0, 'hydrophobicity': -0.9, 'class': 'aromatic'},
    'Y': {'charge': 0, 'hydrophobicity': -1.3, 'class': 'aromatic'},
    'V': {'charge': 0, 'hydrophobicity': 4.2, 'class': 'nonpolar'}
}

def build_mutation_dataset(mapped_scores_path='data/mapped_scores.csv', n_mutations=2000) -> pd.DataFrame:
    """
    Builds mutation dataset.

    Args:
    mapped_scores_path: Location of the mapped scores file. 
    n_mutations: Number of mutations to classify. 

    Returns:
        mut_df: Dataframe containing all mutation classifier variables.
    """

    mapped_scores = pd.read_csv(mapped_scores_path)
    blosum62 = substitution_matrices.load("BLOSUM62")
    amino_acids = list(AA_PROPS.keys())
    mutations = []

    for i in range(n_mutations):
        pos_data = mapped_scores.iloc[np.random.randint(0, len(mapped_scores))]
        wildtype = pos_data['Residue']
        mutant = np.random.choice(amino_acids)

        if wildtype == mutant:
            continue

        wt = AA_PROPS[wildtype]
        mut = AA_PROPS[mutant]

        mutations.append({
            'conservation_score': pos_data['Conservation_Score'],
            'shannon_entropy': pos_data['Shannon_Entropy'],
            'blosum_score': blosum62[(wildtype, mutant)],
            'charge_change': abs(mut['charge'] - wt['charge']),
            'hydrophobicity_change': abs(mut['hydrophobicity'] - wt['hydrophobicity']),
            'class_change': int(mut['class'] != wt['class'])
        })

    mut_df = pd.DataFrame(mutations)
    logging.info(f"Built mutation dataset with {len(mut_df)} mutations")
    return mut_df

def train_gmm(mut_df, n_components=3, random_state=42) -> tuple[GaussianMixture, StandardScaler, dict, pd.DataFrame]:
    """
    Takes in previously extracted variables to train a Gaussian Mixture Model.

    Args:
    mut_df: Dataframe containing all mutation classifier variables.
    n_components: Number of clusters.
    random_state: Initializes pseudo-random number generator to ensure reproducibility.

    Returns:
        gmm: This function does not return a value; it writes output to disk.
        scaler: Scaler model.
        cluster_names: Name of mutation impact categories (clusters).
        mut_df: Final dataframe containing extracted variables and mutation impact predictions. 
    """

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(mut_df)

    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=random_state)
    gmm.fit(x_scaled)

    labels = gmm.predict(x_scaled)
    probs = gmm.predict_proba(x_scaled).max(axis=1)

    mut_df = mut_df.copy()
    mut_df['cluster'] = labels
    mut_df['confidence'] = probs

    # Assign labels by blosum score rank
    mean_blosum = mut_df.groupby('cluster')['blosum_score'].mean().sort_values()
    cluster_names = {
        mean_blosum.index[0]: 'Disruptive',
        mean_blosum.index[1]: 'Moderate',
        mean_blosum.index[2]: 'Neutral'
    }
    mut_df['cluster_label'] = mut_df['cluster'].map(cluster_names)

    logging.info(mut_df['cluster_label'].value_counts().to_string())
    logging.info(f"GMM converged: {gmm.converged_}, Log-likelihood: {gmm.lower_bound_:.3f}")

    return gmm, scaler, cluster_names, mut_df

def plot_cluster_heatmap(mut_df) -> None:
    """
    Creates a heatmap of the mutation impact categories vs. variables to easily visualize patterns.

    Args:
    mut_df: Final dataframe containing extracted variables and mutation impact predictions.

    Returns:
        None: This function does not return a value; it writes output to disk.
    """

    features = ['conservation_score', 'shannon_entropy', 'blosum_score', 'charge_change', 'hydrophobicity_change', 'class_change']
    cluster_means = mut_df.groupby('cluster_label')[features].mean()

    fig, ax = plt.subplots(figsize=(9, 3))
    sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='RdYlGn', center=0, linewidths=0.5, ax=ax)
    ax.set_title('Mean feature value per cluster')
    plt.tight_layout()
    plt.savefig('models/cluster_heatmap.png', dpi=150, bbox_inches='tight')
    logging.info("Saved cluster heatmap to models/cluster_heatmap.png")
    plt.close()

def save_models(gmm, scaler, cluster_names, output_dir='models/') -> None:
    """
    Pickles all machine learning models used.

    Args:
    gmm: Gaussian Mixture Model.
    scalar: Scaler model.
    cluster_names: Name of mutation impact categories (clusters).
    output_dir: Location of the output files.

    Returns:
        None: This function does not return a value; it writes output to disk.
    """

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(gmm,           os.path.join(output_dir, 'gmm.pkl'))
    joblib.dump(scaler,        os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(AA_PROPS,      os.path.join(output_dir, 'aa_properties.pkl'))
    joblib.dump(cluster_names, os.path.join(output_dir, 'cluster_names.pkl'))
    logging.info("Saved models to models/")


