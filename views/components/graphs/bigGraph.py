from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])

Dash_Big_Graph = dcc.Graph(
    id='big-graph',
    figure=fig
)
