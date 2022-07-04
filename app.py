import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from helper.log.LogService import LogService
from jobs.sockets.SocketFactory import SocketFactory
from jobs.PredictSchedule import PredictSchedule
from jobs.WebSocketJob import WebSocketJob
from views.components.graph import Dash_Graph
from views.components.option import Dash_Graph_Option, createMCKOptions
from views.components.header import Dash_Header
from views.callbacks.demo import Demo
import pandas as pd

app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

criterias = {
    "symbol": "XMRBTC",  # sometimes i call it as mck
    "algorithm": "LSTM",
    "features": "Close",
    "isPredict": True,
    "isTrain": True,
}
prediction, dataset = None, None
history = pd.DataFrame()

# # ================================ WEBSOCKET ==================================
websocketJob = WebSocketJob(criterias)


def socket_listener(result):
    tempCriterias = result['criterias']
    if Equals(criterias, tempCriterias):
        criterias['isTrain'] = False
    else:
        websocketJob.runAgain(criterias)
        return

    predictSchedule = PredictSchedule(criterias)
    predictSchedule.addListener(listener)
    predictSchedule.start()


def listener(result):
    global prediction, dataset, criterias, history
    tempCriterias = result['criterias']
    if Equals(criterias, tempCriterias) == True:
        criterias['isPredict'] = False

    prediction = result['prediction']
    dataset = result['dataset']

    last_prediction = prediction.iloc[0]

    if (history.empty == True):
        history = history.append(last_prediction, ignore_index=True)
    else:
        last_history = history.iloc[-1]
        if (last_history['Date'] != last_prediction.Date):
            history = history.append(last_prediction, ignore_index=True)


websocketJob.addListener(socket_listener)
websocketJob.start()

# ================================== Helper Functions ==================================


def Equals(criterias, tempCriterias):
    if tempCriterias['symbol'] != criterias['symbol']:
        return False
    elif tempCriterias['algorithm'] != criterias['algorithm']:
        return False
    elif tempCriterias['features'] != criterias['features']:
        return False
    else:
        return True


# # ================================INITIALIZATION ==================================
websocket = SocketFactory.getSocket()

all_symbols = None


def callback(data):
    global all_symbols
    all_symbols = data


websocket.getAllSymbolTickets(callback)

# ================================ UI AND EVENTS==================================
app.layout = html.Div([
    Dash_Header,
    html.Div([
        html.Div(id='graph-options',
                 children=[
                     Dash_Graph_Option(all_symbols, criterias),
                 ]),
        html.Div(id='live-update-text'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        )
    ])
])


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if criterias['isPredict'] == True | criterias['isTrain'] == True:
        return html.Div(
            [html.Img(src="https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif?20151024034921")])
    return Dash_Graph(dataset, prediction, history)


# Change algorithm &  MCK & feature
@app.callback(
    Output("mck_dropdown", "value"),
    Output("algorithm_dropdown", "value"),
    Output("feature_dropdown", "value"),
    Input('mck_dropdown', 'value'),
    Input('algorithm_dropdown', 'value'),
    Input('feature_dropdown', 'value'))
def update_option(mck, algorithm, features):
    # print(mck, algorithm, features)
    global criterias, websocketJob, history
    if criterias['isTrain'] == True:
        return criterias['symbol'], criterias['algorithm'], criterias['features']

    criterias['symbol'] = mck
    criterias['algorithm'] = algorithm
    criterias['features'] = features

    # Flags
    criterias['isPredict'] = True
    criterias['isTrain'] = True
    # reset history
    history = pd.DataFrame()
    websocketJob.runAgain(criterias)
    return mck, algorithm, features


# Update mck options
@app.callback(
    Output('mck_dropdown', 'options'),
    Input('mck_dropdown', 'value'),)
def update_mck_option(mck):
    return createMCKOptions(all_symbols)


demo = Demo(app)
