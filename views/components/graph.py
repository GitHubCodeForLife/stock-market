from dash import dcc, html
import dash_bootstrap_components as dbc
from views.components.graphs.miniGraph import Dash_Mini_Graph
from views.components.graphs.option import Dash_Graph_Option
from views.components.graphs.bigGraph import Dash_Big_Graph


Dash_Graph = html.Div([
    Dash_Graph_Option,
    html.Br(),
    dbc.Row(
        [
            dbc.Col(Dash_Big_Graph, width=8, sm=12, md=8),
            dbc.Col(Dash_Mini_Graph, width=4, sm=12, md=4),
        ]
    ),
],
    id="graph",
    className="container-fluid",
)
