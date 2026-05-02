import logging
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
    gmm = joblib.load('models/gmm.pkl')
    aa_properties = joblib.load('models/aa_properties.pkl')
    cluster_names = joblib.load('models/cluster_names.pkl')
    blosum62 = substitution_matrices.load("BLOSUM62")
    print("GMM models loaded successfully")
except Exception as e:
    print(f"Could not load models: {e}")
    scaler = gmm = None

# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Load residue position data
def load_conservation_data():
    try:
        df = pd.read_csv("data/mapped_scores.csv")
        return df
    except FileNotFoundError:
        logging.warning("Data/mapped_scores.csv not found.")

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

# Load 3D structure viewer
pdb_dir = './pdb_files'
os.makedirs(pdb_dir, exist_ok=True)
pdb_file = PDBList().retrieve_pdb_file('8JBR', pdir=pdb_dir, file_format='pdb')
dash_parser = DashPdbParser(pdb_file)
pdb_data = dash_parser.mol3d_data()
styles = create_mol3d_style(pdb_data['atoms'], visualization_type='cartoon', color_element='residue')

def load_molecule_viewer():
    return dashbio.Molecule3dViewer(
        id='molecule-3d',
        modelData=pdb_data,
        styles=styles,
        selectionType='atom',
        backgroundColor='#F8F9FA',
        height=410,
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
            html.H1("CyanoStruct: ", className="d-inline text-primary fw-bold", style={'fontSize': '2rem'}),
            html.H1("Mutation Impact Predictor", className="d-inline text-muted", style={'fontSize': '2rem'})
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
                    # Top row: Position + wild-type side by side
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Position:", className="fw-bold mb-1"),
                            dbc.Input(
                                id='position-input',
                                type='number',
                                value=min_pos + 12,
                                min=min_pos,
                                max=max_pos,
                                style={'fontSize': '1.1rem'}
                            ),
                            html.Small(f"Range: {min_pos}-{max_pos}", className="text-muted")
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Wild-Type:", className="fw-bold mb-1"),
                            html.Div(id='wildtype-display',
                                    style={'fontSize': '1.5rem', 'fontWeight': 'bold',
                                           'padding': '6px', 'backgroundColor': '#e9ecef',
                                           'borderRadius': '5px', 'textAlign': 'center'})
                        ], width=6),
                    ], className="mb-3"),

                    # Middle row: Mutant dropdown
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Mutant:", className="fw-bold mb-1"),
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
                                style={'fontSize': '1.1rem'}
                            )
                        ], width=12),
                    # Bottom row: Predict button
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("\u00a0", className="d-block mb-1"),
                            dbc.Button(
                                "Predict Mutation Impact",
                                id='predict-button',
                                color="primary",
                                size="lg",
                                className="w-100",
                            )
                        ], width=12)
                    ])
                    ])
                ], style={'padding': '0.75rem'})
            ], className="shadow-sm", style={'height': '100%'})
        ], width=4), 
 
        # Right column - Prediction results
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
        ], width=8)
    ], className="mb-4", style={'height': '40vh'}),
 
    # Bottom row
    dbc.Row([
        # Feature analysis with plot on left + breakdown on right
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
        
        # Structural view
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Structural View", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    html.Div(id='structural-view', children=[
                        html.Div([
                            load_molecule_viewer()
                        ], style={'display': 'flex',
                                  'justifyContent': 'center',
                                  'alignItems': 'center',
                                  'height': '321px',
                                  'width': '600px',
                                  'margin': '0 auto'})
                    ])
                ], style={'height': 'calc(100% - 45px)', 'overflow': 'hidden', 'padding': '0.5rem'})
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

# Callback to update labelled residue on structural view
@callback(
    Output('molecule-3d', 'labels'),
    Input('position-input', 'value'),
    prevent_initial_call=True
)
def update_molecule_visuals(position):

    if position is None:
        return []

    pdb_index = int(position) - 1269

    atom = next(
        (
            a for a in pdb_data['atoms']
            if int(a['residue_index']) == pdb_index
        ),
        None
    )

    if atom is None:
        return []

    pos = atom['positions']
    residue_name = atom.get('residue_name', '')

    return [{
        'text': f'{residue_name}{position}',
        'position': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
        'fontColor': '#ffffff',
        'font': 'Arial',
        'fontSize': 16,
        'showBackground': True,
        'backgroundColor': '#000000',
    }]

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
        blosum62[wildtype, mutant],
        abs(mut_props['charge'] - wt_props['charge']),
        abs(mut_props['hydrophobicity'] - wt_props['hydrophobicity']),
        int(mut_props['class'] != wt_props['class'])
    ]], columns=['conservation_score','shannon_entropy','blosum_score',
                  'charge_change','hydrophobicity_change','class_change'])
    
    scaled = scaler.transform(features_df)
    cluster = gmm.predict(scaled)[0]
    probs = gmm.predict_proba(scaled)[0]
    confidence = int(probs.max() * 100)

    cluster_names_inv = {v: k for k, v in cluster_names.items()}

    prediction_label = cluster_names.get(cluster, f'Cluster {cluster}').upper()
    color_map = {'DISRUPTIVE': 'danger', 'MODERATE': 'warning', 'NEUTRAL': 'success'}
    prediction_color = color_map.get(prediction_label, 'secondary')

    typicality = gmm.score_samples(scaled)[0] # Higher score = more typical

    # Prediction card
    mutation_string = f"{wildtype}{position}{mutant}"
    prediction_result = dbc.Alert([
    html.H3(f"Prediction: {prediction_label}", className="mb-2 text-white fw-bold"),
    html.P(f"Mutation: {mutation_string}", className="mb-2 text-white fs-6"),
    html.P(f"Confidence: {confidence}%", className="mb-1 text-white fs-5"),
    html.P(f"Typicality Score: {typicality:.2f}", className="mb-2 text-white fs-5"),
    html.Hr(style={'borderColor': 'rgba(255,255,255,0.3)'}),
    html.Div([
        html.Small(f"Neutral: {probs[cluster_names_inv['Neutral']]*100:.0f}%  |  "
                   f"Moderate: {probs[cluster_names_inv['Moderate']]*100:.0f}%  |  "
                   f"Disruptive: {probs[cluster_names_inv['Disruptive']]*100:.0f}%",
                   className="text-white")
    ])
], color=prediction_color, className="text-center py-4")
    
    # Conservation plot
    conservation_plot = dcc.Graph(
        figure=create_conservation_plot(position),
        config={'displayModeBar': False},
        style={'height': '300px'}
    )
    
    # Feature breakdown
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
        html.P(f"Charge Change: {features_df['charge_change'][0]:.1f} {charge_label}", style={'color': charge_color, 'fontWeight':'bold'}),
        html.P(f"AA Class Change: {'Yes' if features_df['class_change'][0] else 'No'}", style={'fontWeight':'bold'}),
        html.P(f"Hydrophobicity Change: {features_df['hydrophobicity_change'][0]:+.1f} ({hyd_label})", style={'fontWeight':'bold'}),
        html.Hr(),
        html.Small("Predictions from trained Gaussian Mixture Model", className="text-muted fst-italic")
    ])
    
    return prediction_result, conservation_plot, feature_breakdown

def create_conservation_plot(mutation_position):
    fig = go.Figure()
    
    # Plot the conservation data
    fig.add_trace(go.Scatter(
        x=conservation_df['PDB_ResNum'],
        y=conservation_df['Conservation_Score'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#007bff', width=1.5),
        fillcolor='rgba(33, 150, 243, 0.15)',
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
        yaxis=dict(range=[0.3, 1])
    )
    
    return fig


# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)