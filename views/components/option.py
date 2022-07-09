from dash import dcc, html
import dash_bootstrap_components as dbc

from views.components.countTime import Dash_CountTime


algorithm_options = [
    {"label": "LSTM", "value": "LSTM"},
    {"label": "RNN", "value": "RNN"},
    {"label": "XGboost", "value": "XGboost"},
    {"label": "Transformer and Time Embeddings",
        "value": "TfmAndTed"},
]

feature_options = [
    {"label": "Close", "value": "Close"},
    {"label": "Price Rate Of Change ", "value": "PROC"},
    {"label": "RSI", "value": "RSI"},
    {"label": "Bolling Bands", "value": "Bolling Bands"},
    {"label": "Exponential Moving Average", "value": "EMA"},
    {"label": "Đường hỗ trợ/kháng cự (nâng cao)",
     "value": "Đường hỗ trợ/kháng cự(nâng cao)"},
]


def createMCKOptions(allSymbols):
    mck_options = []
    for symbol in allSymbols:
        mck_options.append({"label": symbol, "value": symbol})
    return mck_options


def Dash_Graph_Option(all_symbols, criterias):
    mck_options = createMCKOptions(all_symbols)

    return dbc.Row(
        [
            dbc.Col(
                dcc.Dropdown(
                    id="mck_dropdown",
                    options=mck_options,
                    value=criterias['symbol'],
                    # value='XMRBTC',
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
                    value=criterias['algorithm'],
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
                    value=criterias['features'],
                    multi=True,
                    style={"width": "100%"},
                ),
                width=3,
                sm=12,
                md=3,
            ),
            dbc.Col(
                Dash_CountTime()
            ),
        ]
    )
