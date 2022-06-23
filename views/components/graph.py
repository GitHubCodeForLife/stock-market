import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from views.components.graphs.bigGraph import Dash_Big_Graph
from views.components.graphs.miniGraph import Dash_Mini_Graph
from views.components.graphs.option import Dash_Graph_Option

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fig = go.Figure(data=[go.Scatter(x=x, y=y)])
final = 2
mini_fig = go.Figure(data=[go.Scatter(x=x[-final:], y=y[-final:])])


mck_options = [
    {"label": "MCK", "value": "MCK"},
    {"label": "MCD", "value": "MCD"},
    {"label": "MCD_2", "value": "MCD_2"},
    {"label": "MCD_3", "value": "MCD_3"},
    {"label": "MCD_4", "value": "MCD_4"},
    {"label": "MCD_5", "value": "MCD_5"},
    {"label": "MCD_6", "value": "MCD_6"},
]

algorithm_options = [
    {"label": "LSTM", "value": "LSTM"},
    {"label": "RNN", "value": "RNN"},
    {"label": "XGboost", "value": "XGboost"},
    {"label": "Transformer and Time Embeddings",
        "value": "Transformer and Time Embeddings"},
]

feature_options = [
    {"label": "Close", "value": "Close"},
    {"label": "Price Of Change ", "value": "Price Of Change "},
    {"label": "RSI", "value": "RSI"},
    {"label": " Bolling Bands", "value": " Bolling Bands"},
    {"label": "Moving Average", "value": "Moving Average"},
    {"label": "Đường hỗ trợ/kháng cự (nâng cao)",
     "value": "Đường hỗ trợ/kháng cự(nâng cao)"},
]


def Dash_Graph():
    return html.Div([
        Dash_Graph_Option(mck_options, algorithm_options, feature_options),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(Dash_Big_Graph(fig=fig), width=8, sm=12, md=8),
                dbc.Col(Dash_Mini_Graph(fig=mini_fig), width=4, sm=12, md=4),
            ]
        ),
    ],
        id="graph",
        className="container-fluid",
    )
