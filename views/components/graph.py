import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from views.components.graphs.bigGraph import Dash_Big_Graph
from views.components.graphs.miniGraph import Dash_Mini_Graph
from views.components.graphs.option import Dash_Graph_Option


def Dash_Graph(dataset, prediction):

    fig = createBigGraph(dataset, prediction)
    mini_fig = createMiniGraph(dataset, prediction)
    return html.Div([
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


def createBigGraph(dataset, prediction):
    figure = {
        "data": [
            go.Scatter(
                x=prediction['Date'],
                y=prediction["Close"],
                mode='lines',
                fillcolor='red',
                name='Predictions'
            ),
            go.Scatter(
                x=dataset['Date'],
                y=dataset["Close"],
                mode='lines',
                fillcolor='green',
                name='Actual Data'
            )

        ],
        "layout": go.Layout(
            title=" Stock Price Prediction",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'},
        )
    }
    return figure


def createMiniGraph(dataset, prediction):
    final = 10
    return createBigGraph(dataset.tail(final), prediction.head(final))
