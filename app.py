import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from predictor.NSEpredictor import NSEpredictor
from jobs.WebSocketJob import WebSocketJob
from helper.log.LogService import LogService
from jobs.TrainSchedule import TrainSchedule
from jobs.sockets.SocketFactory import SocketFactory

app = dash.Dash()
server = app.server

df = pd.read_csv("./static/data/stock_data.csv")

criterias = {
    "symbol": "xmrbtc",
    "algorithm": "LSTM",
    "features": "Close",
    "isLoadData": True
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
    global valid, train, dataset, criterias
    valid = result['valid']
    train = result['train']
    dataset = result['dataset']
    criterias['isLoadData'] = False


trainSchedule.addListener(listener)
trainSchedule.start()
# ================================INITIALIZATION ==================================
websocket = SocketFactory.getSocket()
all_symbols = None


def callback(data):
    global all_symbols
    all_symbols = data
    LogService().logAppendToFile(str(all_symbols))


websocket.getAllSymbolTickets(callback)

# ================================ UI AND EVENTS==================================
app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[

        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.Div(id="message"),
                dcc.Dropdown(id='my-dropdown',
                             multi=False,
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


@app.callback(Output(component_id='my-dropdown', component_property='options'),
              Input('my-dropdown', 'value'))
def update_figure(selected_dropdown_value):

    global criterias
    # isLoadData = True
    criterias['isLoadData'] = True

    print(selected_dropdown_value)
    LogService().logAppendToFile(
        "Selected Dropdown Value: " + str(selected_dropdown_value))

    if selected_dropdown_value is not None:
        global websocketJob
        criterias["symbol"] = selected_dropdown_value
        websocketJob.runAgain(criterias)

    options = []
    for symbol in all_symbols:
        options.append({'label': symbol, 'value': symbol})

    return options


@ app.callback(Output('live-update-text', 'children'),
               Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if criterias['isLoadData'] == True:
        return "Loading Data..."

    return [dcc.Graph(
        id="Predicted Data",
        figure={
            "data": [
                go.Scatter(
                    x=train.index,
                    y=train["Close"],
                    mode='lines',
                    fillcolor='blue',
                    name='Train Data'
                ),
                go.Scatter(
                    x=valid.index,
                    y=valid["Predictions"],
                    mode='lines',
                    fillcolor='red',
                    name='Predictions'
                ),
                go.Scatter(
                    x=dataset.index,
                    y=dataset["Close"],
                    mode='lines',
                    fillcolor='green',
                    name='Actual Data'
                )

            ],
            "layout":go.Layout(
                title=criterias["symbol"] + " Stock Price Prediction",
                xaxis={'title': 'Date'},
                yaxis={'title': 'Closing Rate'},
            )
        }
    ),
    ]
