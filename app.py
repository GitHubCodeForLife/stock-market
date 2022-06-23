import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from helper.log.LogService import LogService
from jobs.sockets.SocketFactory import SocketFactory
from jobs.TrainSchedule import TrainSchedule
from jobs.WebSocketJob import WebSocketJob
from views.components.graph import Dash_Graph
from views.components.header import Dash_Header

# import plotly.graph_objs as go
# from dash.dependencies import Input, Output


app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP])
server = app.server

# criterias = {
#     "symbol": "XMRBTC",
#     "algorithm": "LSTM",
#     "features": "Close",
#     "isLoadData": True,
#     "isTrain": True,
# }
# tempSymbol = criterias['symbol']
# train, valid, dataset = None, None, None


# # ================================ WEBSOCKET ==================================
# websocketJob = WebSocketJob(criterias)
# websocketJob.start()

# # ================================ TRAIN AND PREDICTOR JOB ==================================
# trainSchedule = TrainSchedule(criterias)


# def listener(result):

#     # print(+ )
#     if result == None:
#         return
#     global valid, train, dataset, criterias
#     valid = result['valid']
#     train = result['train']
#     dataset = result['dataset']
#     # print(valid)
#     if tempSymbol == criterias['symbol']:
#         criterias['isLoadData'] = False
#     else:
#         criterias['symbol'] = tempSymbol


# trainSchedule.addListener(listener)
# trainSchedule.start()


# # ================================INITIALIZATION ==================================
# websocket = SocketFactory.getSocket()
# all_symbols = None


# def callback(data):
#     global all_symbols
#     all_symbols = data
#     # LogService().logAppendToFile(str(all_symbols))


# websocket.getAllSymbolTickets(callback)

# ================================ UI AND EVENTS==================================
app.layout = html.Div([
    Dash_Header,
    Dash_Graph()
])
