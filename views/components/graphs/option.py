from dash import dcc, html
import dash_bootstrap_components as dbc


def Dash_Graph_Option(mck_options, algorithm_options, feature_options):
    return dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id="mck_dropdown",
                    options=mck_options,
                    value=mck_options[0]["value"],
                    style={"width": "100%"},
                ),
                width=3,
                sm=12,
                md=3,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="algorithm_dropdown",
                    options=algorithm_options,
                    value=algorithm_options[0]["value"],
                    style={"width": "100%"},
                ),
                width=3,
                sm=12,
                md=3,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id="feature_dropdown",
                    options=feature_options,
                    value=feature_options[0]["value"],
                    multi=True,
                    style={"width": "100%"},
                ),
                width=3,
                sm=12,
                md=3,
            ),

        ]
    )
