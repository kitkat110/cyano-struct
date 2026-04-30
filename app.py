import os
import dash_bio as dashbio
import dash_bootstrap_components as dbc
import plotly.graph_objects as go 
import plotly.express as px 
from Bio.PDB import PDBList
from dash import Dash, Input, Output, State, callback, html, dcc
from dash_bio.utils import PdbParser as DashPdbParser
from dash_bio.utils import create_mol3d_style
import pandas as pd
import numpy as np

# ML model imports
import joblib
from Bio.Align import substitution_matrices

# Load ML model components
try:
    scaler = joblib.load('models/scaler.pkl')
    pca = joblib.load('models/pca.pkl')
    kmeans = joblib.load('models/kmeans.pkl')
    aa_properties = joblib.load('models/aa_properties.pkl')
    blosum62 = substitution_matrices.load("BLOSUM62")
    print("✅ Unsupervised models loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load unsupervised models: {e}")
    scaler = pca = kmeans = None

# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Load conservation data
def load_conservation_data():
    try:
        df = pd.read_csv("data/mapped_scores.csv")
        return df
    except FileNotFoundError:
        print("Warning: data/mapped_scores.csv not found.")

conservation_df = load_conservation_data()
print(f"Loaded conservation data for {len(conservation_df)} residues")

def get_residue_data_at_position(position):
    residue_data = conservation_df[conservation_df['PDB_ResNum'] == position]
    
    if not residue_data.empty and 'Residue' in residue_data.columns:
        residue_num = residue_data['PDB_ResNum'].iloc[0]
        residue = residue_data['Residue'].iloc[0]
        conservation = residue_data['Conservation_Score'].iloc[0]
        entropy = residue_data['Shannon_Entropy'].iloc[0]
    else:
        print("Residue position has no available data.") 
    return (residue_num, residue, conservation, entropy)

# Helper function for BLOSUM score
def get_blosum_score(aa1, aa2):
    try:
        return blosum62[(aa1, aa2)]
    except:
        return 0

# Real ML prediction function 
def predict_mutation_cluster(position, wildtype, mutant, conservation_score, shannon_entropy):
    """Predict mutation cluster using scaler → PCA → KMeans"""
    wt = aa_properties.get(wildtype, aa_properties['A'])
    mut = aa_properties.get(mutant, aa_properties['A'])

    # Features in same order as during training
    feature_cols = ['conservation_score', 'shannon_entropy', 'blosum_score',
                'charge_change', 'hydrophobicity_change', 'class_change']

    # Compute features in same order as training
    features = pd.DataFrame([[
        conservation_score,
        shannon_entropy,
        get_blosum_score(wildtype, mutant),
        abs(mut['charge'] - wt['charge']),
        abs(mut['hydrophobicity'] - wt['hydrophobicity']),
        int(mut['class'] != wt['class'])
    ]], columns=feature_cols)

    # Apply the pipeline: scaler -> PCA -> KMeans
    scaled = scaler.transform(features)
    reduced = pca.transform(scaled)
    cluster = kmeans.predict(reduced)[0]

    # Optional: map cluster numbers to labels
    cluster_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    cluster_label = cluster_labels.get(cluster, f"Cluster {cluster}")

    return {
        'cluster': cluster_label,
        'cluster_id': cluster,
        'reduced_features': reduced[0],
        'blosum_score': get_blosum_score(wildtype, mutant),
        'charge_change': abs(mut['charge'] - wt['charge']),
        'hydrophobicity_change': mut['hydrophobicity'] - wt['hydrophobicity'],
        'class_change': mut['class'] != wt['class'],
        'wt_class': wt['class'],
        'mut_class': mut['class']
    }

# Load 3D structure viewer
def load_molecule_viewer():
    pdb_id = '8JBR'

    pdb_dir = './pdb_files'
    os.makedirs(pdb_dir, exist_ok=True)

    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')

    dash_parser = DashPdbParser(pdb_file)
    pdb_data = dash_parser.mol3d_data()

    styles = create_mol3d_style(
        pdb_data['atoms'],
        visualization_type='cartoon',
        color_element='residue'
    )

    return dashbio.Molecule3dViewer(
        id='molecule-3d',
        modelData=pdb_data,
        styles=styles,
        selectionType='atom',
        backgroundColor='#F8F9FA',
        height=250,
        width='100%'
    )

# Get position range from data
min_pos = int(conservation_df['PDB_ResNum'].min())
max_pos = int(conservation_df['PDB_ResNum'].max())

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        html.Div([
            html.H1("CyanoStruct: ", className="d-inline text-primary fw-bold", style={'fontSize': '1.8rem'}),
            html.H1("Mutation Impact Predictor", className="d-inline text-muted", style={'fontSize': '1.8rem'}),
            html.Small(className="d-block text-center mt-2",
                       style={'color': '#28a745' if kmeans else '#ffc107'})
        ], className="text-center mb-1 py-1", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ], className="mb-1"),
 
    # Main content
    dbc.Row([
        # Left column - Input
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Mutation Input", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    # Position input
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Position:", className="fw-bold mb-2"),
                            dbc.Input(
                                id='position-input',
                                type='number',
                                value=min_pos + 12,  # Default to position 1281 (D->K example)
                                min=min_pos,
                                max=max_pos,
                                className='mb-3',
                                style={'fontSize': '1.1rem'}
                            ),
                            html.Small(f"Range: {min_pos}-{max_pos}", 
                                     className="text-muted")
                        ])
                    ]),
                    
                    # Wild-type display (auto-populated from your data)
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Wild-Type:", className="fw-bold mb-2"),
                            html.Div(id='wildtype-display', 
                                   style={'fontSize': '1.2rem', 'fontWeight': 'bold', 
                                         'padding': '4px', 'backgroundColor': '#e9ecef',
                                         'borderRadius': '5px', 'textAlign': 'center'})
                        ])
                    ], className="mb-1"),
                    
                    # Mutant input
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mutant:", className="fw-bold mb-2"),
                            dbc.Select(
                                id='mutant-input',
                                options=[
                                    {'label': f'{aa} - {name}', 'value': aa} 
                                    for aa, name in [
                                        ('A', 'Alanine'), ('R', 'Arginine'), ('N', 'Asparagine'), ('D', 'Aspartic acid'),
                                        ('C', 'Cysteine'), ('Q', 'Glutamine'), ('E', 'Glutamic acid'), ('G', 'Glycine'),
                                        ('H', 'Histidine'), ('I', 'Isoleucine'), ('L', 'Leucine'), ('K', 'Lysine'),
                                        ('M', 'Methionine'), ('F', 'Phenylalanine'), ('P', 'Proline'), ('S', 'Serine'),
                                        ('T', 'Threonine'), ('W', 'Tryptophan'), ('Y', 'Tyrosine'), ('V', 'Valine')
                                    ]
                                ],
                                value='K',
                                className='mb-4',
                                style={'fontSize': '1.1rem'}
                            )
                        ])
                    ]),
                    
                    # Predict button
                    dbc.Button(
                        "Predict Mutation Impact",
                        id='predict-button',
                        color="primary",
                        size="lg",
                        className="w-100",
                        style={'padding': '12px'}
                    )
                ])
            ], className="shadow-sm", style={'height': '100%'})
        ], width=3),
 
        # Right column - Prediction Result
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Prediction Result", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    html.Div(id='prediction-result', children=[
                        html.Div([
                            html.P("Select position and mutant amino acid, then click 'Predict Mutation Impact'", 
                                  className="text-center text-muted mt-4")
                        ])
                    ])
                ])
            ], className="shadow-sm", style={'height': '100%'})
        ], width=9)
    ], className="mb-4", style={'height': '50vh'}),
 
    # Bottom row
    dbc.Row([
        # Feature Analysis with plot on left, breakdown on right
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Feature Breakdown", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        # Conservation plot on the left
                        dbc.Col([
                            html.Div(id='conservation-plot', children=[
                                html.P("Conservation plot will appear here", 
                                      className="text-center text-muted mt-5")
                            ])
                        ], width=6),
                        
                        # Feature list on the right
                        dbc.Col([
                            html.Div(id='feature-breakdown', children=[
                                html.P("Feature analysis will appear here after prediction", 
                                      className="text-center text-muted mt-4")
                            ])
                        ], width=6)
                    ])
                ], style={'height': 'calc(100% - 45px)', 'overflowY': 'auto'})
            ], className="shadow-sm", style={'height': '100%', 'overflow': 'hidden'})
        ], width=6, style={'height': '100%'}),
        
        # Structural View
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Structural View", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    html.Div(id='structural-view', children=[
                        html.Div([
                            load_molecule_viewer()
                        ], style={'height': '300px'}),
                        html.Hr()
                    ])
                ], style={'height': 'calc(100% - 45px)', 'overflowY': 'auto'})
            ], className="shadow-sm", style={'height': '100%', 'overflow': 'hidden'})
        ], width=6, style={'height': '100%'})
    ])
], fluid=True, style={'flex': '1 1 0', 'minHeight': '0', 'overflow': 'hidden'})

# Callback to update wild-type display
@callback(
    Output('wildtype-display', 'children'),
    Input('position-input', 'value')
)
def update_wildtype_display(position):
    if position is None:
        return "Enter position"
    
    residue = get_residue_data_at_position(position)[1]
    return residue

# Callback to load and display molecule
@callback(
    [Output('prediction-result', 'children'),
     Output('conservation-plot', 'children'),
     Output('feature-breakdown', 'children')],
    Input('predict-button', 'n_clicks'),
    [State('position-input', 'value'),
     State('mutant-input', 'value')],
    prevent_initial_call=True
)
def update_prediction(n_clicks, position, mutant):
    if n_clicks is None or position is None:
        return "", "", ""
    
    # Get wildtype and residue info
    residue_info = get_residue_data_at_position(position)
    wildtype = residue_info[1]
    conservation_score = residue_info[2]
    shannon_entropy = residue_info[3]

    # Construct feature vector
    wt_props = aa_properties.get(wildtype, aa_properties['A'])
    mut_props = aa_properties.get(mutant, aa_properties['A'])
    
    features_df = pd.DataFrame([[
        conservation_score,
        shannon_entropy,
        get_blosum_score(wildtype, mutant),
        abs(mut_props['charge'] - wt_props['charge']),
        abs(mut_props['hydrophobicity'] - wt_props['hydrophobicity']),
        int(mut_props['class'] != wt_props['class'])
    ]], columns=['conservation_score','shannon_entropy','blosum_score',
                  'charge_change','hydrophobicity_change','class_change'])
    
    # Scale and predict KMeans cluster
    scaled = scaler.transform(features_df)
    cluster = kmeans.predict(scaled)[0]

    # Assign a label based on cluster (example: 0=Neutral, 1=Disruptive)
    prediction_label = "DISRUPTIVE" if cluster == 1 else "NEUTRAL"
    prediction_color = "danger" if prediction_label == 'DISRUPTIVE' else "success"
    
    # Risk/confidence score as a mock from distance to cluster center
    risk_score = np.linalg.norm(scaled - kmeans.cluster_centers_[cluster])
    confidence = int(max(0.6, 1 - risk_score) * 100)  # min 60% confidence
    
    # Prediction card
    mutation_string = f"{wildtype}{position}{mutant}"
    prediction_result = dbc.Alert([
        html.H3(f"Prediction: {prediction_label}", className="mb-2 text-white fw-bold"),
        html.P(f"Mutation: {mutation_string}", className="mb-2 text-white fs-6"),
        html.P(f"Confidence: {confidence}%", className="mb-1 text-white fs-5"),
        html.P(f"Risk Score: {risk_score:.2f}", className="mb-0 text-white fs-5")
    ], color=prediction_color, className="text-center py-4")
    
    # Conservation plot
    conservation_plot = dcc.Graph(
        figure=create_conservation_plot(position),
        config={'displayModeBar': False},
        style={'height': '300px'}
    )
    
    # Feature breakdown (styled like your screenshot)
    def style_level(value, thresholds):
        if value >= thresholds[0]:
            return "(High)", "#28a745"
        elif value >= thresholds[1]:
            return "(Medium)", "#ffc107"
        else:
            return "(Low)", "#dc3545"
    
    cons_label, cons_color = style_level(conservation_score, [0.9, 0.7])
    entropy_label, entropy_color = style_level(shannon_entropy, [0.5, 1.0])
    blosum_label, blosum_color = ("Favorable", "#28a745") if features_df['blosum_score'][0]>=0 else ("Unfavorable", "#dc3545")
    charge_label, charge_color = style_level(features_df['charge_change'][0], [1,0.5])
    hyd_label = "More hydrophobic" if features_df['hydrophobicity_change'][0] > 0 else "Less hydrophobic" if features_df['hydrophobicity_change'][0] < 0 else "No change"

    feature_breakdown = html.Div([
        html.P(f"Conservation Score: {conservation_score:.3f} {cons_label}", style={'color': cons_color, 'fontWeight':'bold'}),
        html.P(f"Shannon Entropy: {shannon_entropy:.3f} {entropy_label}", style={'color': entropy_color, 'fontWeight':'bold'}),
        html.P(f"BLOSUM62 Score: {features_df['blosum_score'][0]} ({blosum_label})", style={'color': blosum_color, 'fontWeight':'bold'}),
        html.P(f"Charge Change: {features_df['charge_change'][0]:.1f} ({charge_label})", style={'color': charge_color, 'fontWeight':'bold'}),
        html.P(f"AA Class Change: {'Yes' if features_df['class_change'][0] else 'No'}", style={'fontWeight':'bold'}),
        html.P(f"Hydrophobicity Change: {features_df['hydrophobicity_change'][0]:+.1f} ({hyd_label})", style={'fontWeight':'bold'}),
        html.Hr(),
        html.Small("Predictions from trained KMeans model", className="text-muted fst-italic")
    ])
    
    return prediction_result, conservation_plot, feature_breakdown

def create_conservation_plot(mutation_position):
    fig = go.Figure()
    
    # Plot your actual conservation data
    fig.add_trace(go.Scatter(
        x=conservation_df['PDB_ResNum'],
        y=conservation_df['Conservation_Score'],
        mode='lines',
        line=dict(color='#007bff', width=2),
        name='Conservation Score',
        hovertemplate='Position: %{x}<br>Conservation: %{y:.3f}<br>Residue: %{text}<extra></extra>',
        text=conservation_df['Residue'] if 'Residue' in conservation_df.columns else [''] * len(conservation_df)
    ))
    
    # Highlight mutation position
    mutation_data = conservation_df[conservation_df['PDB_ResNum'] == mutation_position]
    if not mutation_data.empty:
        fig.add_trace(go.Scatter(
            x=[mutation_position],
            y=[mutation_data['Conservation_Score'].iloc[0]],
            mode='markers',
            marker=dict(color='red', size=12, symbol='circle'),
            name='Mutation Site',
            hovertemplate=f'Mutation: {mutation_position}<br>Conservation: {mutation_data["Conservation_Score"].iloc[0]:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text="Conservation Score Along Sequence", font_size=14, x=0.5),
        xaxis_title="Residue Position",
        yaxis_title="Conservation Score",
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor='rgba(248,249,250,0.8)',
        yaxis=dict(range=[0, 1])
    )
    
    return fig


# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)