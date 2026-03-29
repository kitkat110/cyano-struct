import os

import dash_bio as dashbio
import dash_bootstrap_components as dbc
from Bio.PDB import PDBList
from dash import Dash, Input, Output, State, callback, html
from dash_bio.utils import PdbParser as DashPdbParser
from dash_bio.utils import create_mol3d_style

# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div(
            "CyanoStruct: Cyanobacterial Toxin Protein Mutation Impact Predictor",
            className="text-primary text-center fs-3 mb-4"
        )
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label("Position", className="fw-bold"),
            dbc.Input(
                id='mutation-input',
                type='text',
                value='1269',
                className='mb-2'
            ),
            dbc.Button("Predict Mutation Impact", id='load-button', color="primary"),
            html.Div(id='status-message', className='mt-3')
        ], width=2),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(
                id='molecular-viewer',
                children=[
                    html.Div(
                        "Enter a position, wild-type, and mutant amino acid and click 'Predict Mutation Impact' to view the molecule.",
                        className="text-center text-muted mt-5"
                    )
                ]
            )
        ], width=10)
    ])
])

# Callback to load and display molecule
@callback(
    [Output('molecular-viewer', 'children'),
    Output('status-message', 'children')],
    Input('load-button', 'n_clicks'),
    State('mutation-input', 'value'),
    prevent_initial_call=True
)


def load_molecule(load_clicks, mutation_pos):
    pdb_id = '8JBR'

    pdb_dir = './pdb_files'
    os.makedirs(pdb_dir, exist_ok=True)

    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format='pdb')

    dash_parser = DashPdbParser(pdb_file)
    pdb_data = dash_parser.mol3d_data()

    styles = create_mol3d_style(
        pdb_data['atoms'], visualization_type='cartoon', color_element='residue'
    )

    # Create Molecule3dViewer component
    viewer = create_molecule_viewer(pdb_data, styles)

    status = dbc.Alert(
        f"Successfully loaded PDB ID: {pdb_id.upper()}",
        color="success"
    )

    return viewer, status 

def create_molecule_viewer(pdb_data, styles):
    """Create a Molecule3dViewer from PDB data"""
    return dashbio.Molecule3dViewer(
        id='molecule-3d',
        modelData=pdb_data,
        styles=styles,
        selectionType='atom',
        backgroundColor='#F0F0F0',
        height=600,
        width='100%'
    )

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True)