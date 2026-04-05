import numpy as np
import joblib
from Bio.Align import substitution_matrices

model = joblib.load('models/random_forest_model.pkl')
features = joblib.load('models/feature_names.pkl')
aa_props = joblib.load('models/aa_properties.pkl')
blosum62 = substitution_matrices.load("BLOSUM62")

def get_blosum_score(aa1, aa2):
    try:
        return blosum62[(aa1, aa2)]
    except:
        return 0

def predict_mutation_impact(position, wildtype, mutant, conservation_score, shannon_entropy):
    wt = aa_props.get(wildtype, aa_props['A'])
    mut = aa_props.get(mutant, aa_props['A'])

    mutation_features = [[
        conservation_score,
        shannon_entropy,
        get_blosum_score(wildtype, mutant),
        abs(mut['charge'] - wt['charge']),
        abs(mut['hydrophobic'] - wt['hydrophobic']),
        int(mut['class'] != wt['class'])
    ]]

    prediction_proba = model.predict_proba(mutation_features)[0, 1]
    prediction = int(prediction_proba > 0.5)
    confidence = int(max(prediction_proba, 1 - prediction_proba) * 100)

    return {
        'prediction': 'DISRUPTIVE' if prediction == 1 else 'NEUTRAL',
        'confidence': max(confidence, 60),
        'risk_score': prediction_proba,
        'blosum_score': get_blosum_score(wildtype, mutant),
        'charge_change': abs(mut['charge'] - wt['charge']),
        'hydrophobic_change': mut['hydrophobic'] - wt['hydrophobic'],
        'class_change': mut['class'] != wt['class'],
        'wt_class': wt['class'],
        'mut_class': mut['class']
    }
