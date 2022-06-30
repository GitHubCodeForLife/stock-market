from dash import dcc, html
import dash_bootstrap_components as dbc


def Dash_CountTime():
    return html.Div(
        children=[
            html.Div(
                id="count_time",
            ),
        ],
    )
