from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

Dash_Mini_Graph = dcc.Graph(
    id='mini-graph',
    figure=fig
)
