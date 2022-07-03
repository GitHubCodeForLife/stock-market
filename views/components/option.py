from dash import dcc, html
import dash_bootstrap_components as dbc

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


def createMCKOptions(allSymbols):
    mck_options = []
    for symbol in allSymbols:
        mck_options.append({"label": symbol, "value": symbol})
    return mck_options


def Dash_Graph_Option(all_symbols):
    mck_options = createMCKOptions(all_symbols)

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
