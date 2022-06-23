from dash import dcc, html


Dash_Header = html.Div(
    [
        html.H2(
            "Live Model Training Viewer",
            id="title",
            className="eight columns",
            style={"margin-left": "3%"},
        ),
        html.Br(),
    ],
    className="banner row",
)
