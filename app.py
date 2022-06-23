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
from views.components.graphs.option import Dash_Graph_Option, createMCKOptions
from views.components.header import Dash_Header

app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP])
server = app.server

criterias = {
    "symbol": "XMRBTC",
    "algorithm": "LSTM",
    "features": "Close",
    "isPredict": True,
    "isTrain": True,
}
train, valid, dataset = None, None, None


# # ================================ WEBSOCKET ==================================
websocketJob = WebSocketJob(criterias)


def socket_listener(result):
    # print("socket_listener")
    # print(result)
    tempCriterias = result['criterias']
    if Equals(criterias, tempCriterias):
        criterias['isTrain'] = False


websocketJob.addListener(socket_listener)
websocketJob.start()

# ================================ PREDICTOR JOB ==================================
predictSchedule = PredictSchedule(criterias)


def listener(result):
    if result == None:
        return
    global valid, train, dataset, criterias
    tempCriterias = result['criterias']
    if Equals(criterias, tempCriterias) == True:
        criterias['isPredict'] = False

    valid = result['valid']
    train = result['train']
    dataset = result['dataset']


predictSchedule.addListener(listener)
predictSchedule.start()

# ================================== Helper ==================================


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
                     Dash_Graph_Option(all_symbols),
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
    return Dash_Graph(dataset, valid)


# Change algorithm &  MCK & feature
@app.callback(
    Output("indicator", "value"),
    Input('mck_dropdown', 'value'),
    Input('algorithm_dropdown', 'value'),
    Input('feature_dropdown', 'value'))
def update_option(mck, algorithm, features):
    print(mck, algorithm, features)
    global criterias, websocketJob, predictSchedule
    if criterias['isTrain'] == True:
        return ""

    criterias['symbol'] = mck
    criterias['algorithm'] = algorithm
    criterias['features'] = features

    # Flags
    criterias['isPredict'] = True
    criterias['isTrain'] = True

    websocketJob.runAgain(criterias)
    predictSchedule.setCriterias(criterias)
    return ""


# Update mck options
@app.callback(
    Output('mck_dropdown', 'options'),
    Input('mck_dropdown', 'value'),)
def update_mck_option(mck):
    return createMCKOptions(all_symbols)
