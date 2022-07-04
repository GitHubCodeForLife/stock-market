from typing import final
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from helper.log.LogService import LogService

from views.components.graphs.bigGraph import Dash_Big_Graph
from views.components.graphs.miniGraph import Dash_Mini_Graph


def Dash_Graph(dataset, prediction, history):
    final = 10
    fig = createBigFigure(dataset, prediction, history)

    mini_fig = createMiniFigure(dataset.tail(
        final), prediction.head(final), history.tail(final+1))
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


def createBigFigure(dataset, prediction, history):

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
            ),
            go.Scatter(
                x=history['Date'],
                y=history["Close"],
                mode='lines',
                fillcolor='blue',
                name='History Predict'
            )

        ],
        "layout": go.Layout(
            title=" Stock Price Prediction",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'},
        )
    }
    return figure


def createMiniFigure(dataset, prediction, history):
    figure = {
        "data": [
            go.Scatter(
                x=prediction['Date'],
                y=prediction["Close"],
                mode='markers',
                fillcolor='red',
                name='Predictions',
            ),
            go.Scatter(
                x=dataset['Date'],
                y=dataset["Close"],
                mode='lines',
                fillcolor='green',
                name='Actual Data',
            ),
            go.Scatter(
                x=history['Date'],
                y=history["Close"],
                mode='lines',
                fillcolor='blue',
                name='History Predict',
            )
        ],
        "layout": go.Layout(
            title=" Stock Price Prediction",
            xaxis={'title': 'Date'},
            yaxis={'title': 'Closing Rate'},

        )
    }
    return figure
