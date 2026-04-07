from dash import Dash, html
import dash_bio as dashbio

app = Dash(__name__)

with open('data/aligned_microcystis_sequences.fasta', 'r') as f:
    data = f.read()

app.layout = html.Div([
    dashbio.AlignmentChart(
        id='alignment-viewer',
        data=data,
        showconservation=False,
        showgap=False,
        height=900,
        tilewidth=30
    ),
])

if __name__ == '__main__':
    app.run(debug=True)