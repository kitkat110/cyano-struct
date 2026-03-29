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

# Mock ML prediction function (placeholder for actual ML model)
def mock_ml_prediction(position, wildtype, mutant, conservation_score, shannon_entropy):
    
    # Simple heuristic for demo - replace with actual ML prediction
    risk_score = 0.7 * conservation_score + 0.2 * (2.0 - shannon_entropy) + 0.1 * np.random.random()
    confidence = min(int(risk_score * 100 + np.random.randint(10, 25)), 95)
    
    # Mock other features for display
    mock_features = {
        'blosum_score': np.random.randint(-3, 3),
        'hydrophobic_change': np.random.uniform(-4, 4),
        'charge_change': np.random.choice([0, 1, 2]),
        'class_change': np.random.choice([True, False])
    }
    
    return {
        'prediction': 'DISRUPTIVE' if risk_score > 0.6 else 'NEUTRAL',
        'confidence': confidence,
        'risk_score': min(risk_score, 1.0),
        **mock_features
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
            html.H1("CyanoStruct: ", className="d-inline text-primary fw-bold", style={'fontSize': '2.5rem'}),
            html.H1("Mutation Impact Predictor", className="d-inline text-muted", style={'fontSize': '2.5rem'})
        ], className="text-center mb-4 py-3", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ], className="mb-4"),
 
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
                                   style={'fontSize': '1.5rem', 'fontWeight': 'bold', 
                                         'padding': '10px', 'backgroundColor': '#e9ecef',
                                         'borderRadius': '5px', 'textAlign': 'center'})
                        ])
                    ], className="mb-3"),
                    
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
            ], className="shadow-sm")
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
            ], className="shadow-sm")
        ], width=3)
    ], className="mb-4"),
 
    # Analysis Section
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
                ])
            ], className="shadow-sm")
        ], width=6),
        
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
                        html.Hr(),
                        html.Div([
                            html.P([
                                html.Span("Distance to Active Site: ", className="fw-bold")
                            ], className="mb-1"),
                            html.P([
                                html.Span("Solvent Accessibility: ", className="fw-bold")
                            ], className="mb-0")
                        ], className="text-center")
                    ])
                ])
            ], className="shadow-sm")
        ], width=6)
    ])
])

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

    wildtype = get_residue_data_at_position(position)[1]
    conservation_score = get_residue_data_at_position(position)[2]
    shannon_entropy = get_residue_data_at_position(position)[3]

    prediction_data = mock_ml_prediction(position, wildtype, mutant, conservation_score, shannon_entropy)

    mutation_string = f"{wildtype}{position}{mutant}"

    # Prediction result
    prediction_color = "danger" if prediction_data['prediction'] == 'DISRUPTIVE' else "success"
    prediction_result = html.Div([
        dbc.Alert([
            html.H3(f"Prediction: {prediction_data['prediction']}", 
                   className="mb-2 text-white fw-bold"),
            html.P(f"Mutation: {mutation_string}", 
                  className="mb-2 text-white fs-6"),
            html.P(f"Confidence: {prediction_data['confidence']}%", 
                  className="mb-1 text-white fs-5"),
            html.P(f"Risk Score: {prediction_data['risk_score']:.2f}", 
                  className="mb-0 text-white fs-5")
        ], color=prediction_color, className="text-center py-4")
    ])

    # Conservation plot
    conservation_plot = dcc.Graph(
        figure=create_conservation_plot(position),
        config={'displayModeBar': False},
        style={'height': '300px'}
    )

    # Feature breakdown list 
    feature_breakdown = html.Div([
        # Conservation Score
        html.Div([
            html.I(className="bi bi-circle-fill me-2", style={'color': '#17a2b8'}),
            html.Span("Conservation Score: ", className="fw-bold"),
            html.Span(f"{conservation_score:.3f} ", className="fw-bold"),
            html.Span("(High)" if conservation_score >= 0.9 else "(Medium)" if conservation_score >= 0.7 else "(Low)", 
                     style={'color': '#28a745' if conservation_score >= 0.9 else '#ffc107' if conservation_score >= 0.7 else '#dc3545'})
        ], className="mb-3"),
        
        # Shannon Entropy
        html.Div([
            html.I(className="bi bi-circle-fill me-2", style={'color': '#17a2b8'}),
            html.Span("Shannon Entropy: ", className="fw-bold"),
            html.Span(f"{shannon_entropy:.3f} ", className="fw-bold"),
            html.Span("(Low)" if shannon_entropy < 0.5 else "(Medium)" if shannon_entropy < 1.0 else "(High)", 
                     style={'color': '#28a745' if shannon_entropy < 0.5 else '#ffc107' if shannon_entropy < 1.0 else '#dc3545'})
        ], className="mb-3"),
        
        # Mock features (placeholders for future ML features)
        html.Div([
            html.I(className="bi bi-circle-fill me-2", style={'color': '#6f42c1'}),
            html.Span("BLOSUM Score: ", className="fw-bold"),
            html.Span(f"{prediction_data['blosum_score']} ", className="fw-bold"),
            html.Span("(Unfavorable)" if prediction_data['blosum_score'] < 0 else "(Favorable)", 
                     style={'color': '#dc3545' if prediction_data['blosum_score'] < 0 else '#28a745'})
        ], className="mb-3"),
        
        html.Div([
            html.I(className="bi bi-circle-fill me-2", style={'color': '#fd7e14'}),
            html.Span("AA Class Change: ", className="fw-bold"),
            html.Span("Yes " if prediction_data['class_change'] else "No ", className="fw-bold"),
            html.Span("(Negative to Positive)" if prediction_data['class_change'] else "(Same Class)", 
                     style={'color': '#6c757d'})
        ], className="mb-3"),
        
        html.Div([
            html.I(className="bi bi-circle-fill me-2", style={'color': '#28a745'}),
            html.Span("Hydrophobic Change: ", className="fw-bold"),
            html.Span(f"{prediction_data['hydrophobic_change']:+.1f}", className="fw-bold")
        ], className="mb-3")
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