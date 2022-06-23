from dash import dcc, html
import dash_bootstrap_components as dbc


def Dash_Mini_Graph(fig):
    return dcc.Graph(
        id='mini-graph',
        figure=fig
    )
