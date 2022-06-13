import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from predictor.NSEpredictor import NSEpredictor
from modeltrainer.LSTMTrainer import LSTMTrainer
from jobs.WebSocketJob import WebSocketJob
from helper.log.LogService import LogService
from jobs.TrainSchedule import TrainSchedule

app = dash.Dash()
server = app.server


df = pd.read_csv("./static/data/stock_data.csv")


criterias = {
    "symbol": "xmrbtc",
    "algorithm": "LSTM",
    "features": "Close"
}


# ================================ PREDICTOR ==================================
# lsmtTrainer = LSTMTrainer()
# lsmtTrainer.run()

nseTrainer = NSEpredictor()
train, valid, dataset = nseTrainer.run()

# ================================ WEBSOCKET ==================================
websocketJob = WebSocketJob(criterias)
websocketJob.start()

# ================================ TRAIN AND PREDICTOR JOB ==================================
trainSchedule = TrainSchedule(criterias)


def listener(result):
    global valid, train, dataset
    valid = result['valid']
    train = result['train']
    dataset = result['dataset']
    LogService().logAppendToFile("listener")
    LogService().logAppendToFile(str(valid))
    LogService().logAppendToFile(str(train))
    LogService().logAppendToFile(str(dataset))


trainSchedule.addListener(listener)
trainSchedule.start()

# ================================ UI AND EVENTS==================================
app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                html.Div([
                    html.H4('TERRA Satellite Live Feed'),
                    html.Div(id='live-update-text'),
                    dcc.Interval(
                        id='interval-component',
                        interval=1*1000,  # in milliseconds
                        n_intervals=0
                    )
                ])
            ])
        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows",
                        style={'textAlign': 'center'}),


                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_figure(selected_dropdown_value):
    print(selected_dropdown_value)
    global websocketJob, criterias
    criterias["symbol"] = "btcusdt"
    websocketJob.runAgain(criterias)


@ app.callback(Output('live-update-text', 'children'),
               Input('interval-component', 'n_intervals'))
def update_metrics(n):

    return [dcc.Graph(
        id="Predicted Data",
        figure={
            "data": [
                go.Scatter(
                    x=train.index,
                    y=train["Close"],
                    mode='lines',
                    fillcolor='blue',
                ),
                go.Scatter(
                    x=valid.index,
                    y=valid["Predictions"],
                    mode='lines',
                    fillcolor='red',
                ),
                go.Scatter(
                    x=dataset.index,
                    y=dataset["Close"],
                    mode='lines',
                    fillcolor='green',
                )

            ],
            "layout":go.Layout(
                title='scatter plot',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Closing Rate'},
            )
        }
    ),
    ]
