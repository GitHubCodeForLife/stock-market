from dash import dcc, html
import dash_bootstrap_components as dbc


mck_options = [
    {"label": "MCK", "value": "MCK"},
    {"label": "MCD", "value": "MCD"},
    {"label": "MCD_2", "value": "MCD_2"},
    {"label": "MCD_3", "value": "MCD_3"},
    {"label": "MCD_4", "value": "MCD_4"},
    {"label": "MCD_5", "value": "MCD_5"},
    {"label": "MCD_6", "value": "MCD_6"},
]

feature_options = [
    {"label": "Feature 1", "value": "Feature 1"},
    {"label": "Feature 2", "value": "Feature 2"},
    {"label": "Feature 3", "value": "Feature 3"},
    {"label": "Feature 4", "value": "Feature 4"},
    {"label": "Feature 5", "value": "Feature 5"},
]


Dash_Graph_Option = dbc.Row(
    [
        dbc.Col(
            dcc.Dropdown(
                id="mck_dropdown",
                options=mck_options,
                value="MCK",
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
                value="Feature 1",
                multi=True,
                style={"width": "100%"},
            ),
            width=3,
            sm=12,
            md=3,
        ),
    ]
)
