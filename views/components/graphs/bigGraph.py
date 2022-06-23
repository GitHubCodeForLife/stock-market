from dash import dcc, html
import dash_bootstrap_components as dbc


def Dash_Big_Graph(fig):
    return dcc.Graph(
        id='big-graph',
        figure=fig
    )
