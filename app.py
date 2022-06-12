from datetime import datetime
from turtle import color
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
from predictor.NSEpredictor import NSEpredictor
from modeltrainer.LSTMTrainer import LSTMTrainer
from jobs.WebSocketJob import WebSocketJob
app = dash.Dash()
server = app.server
# lsmtTrainer = LSTMTrainer()
# lsmtTrainer.run()

nseTrainer = NSEpredictor()
nseTrainer.run()
train = nseTrainer.train
valid = nseTrainer.valid


test = 0
websocketJob = WebSocketJob()


def callback(result):
    # print(result)
    # result = json.loads(result)
    global valid, train
    global websocketJob
    valid = result["valid"]
    train = result["train"]
    # websocketJob.setSymbol("TSLA")


websocketJob.addListener(callback)
websocketJob.start()

df = pd.read_csv("./static/data/stock_data.csv")

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
    global websocketJob
    websocketJob.setSymbol(selected_dropdown_value[0])


@ app.callback(Output('live-update-text', 'children'),
               Input('interval-component', 'n_intervals'))
def update_metrics(n):

  # log to file txt
    file = "./static/data/data1.txt"
    with open(file, "w") as f:
        f.write(str(valid))
        f.write("\n")
        f.write(str(train))

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
            ],
            "layout":go.Layout(
                title='scatter plot',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Closing Rate'},
            )
        }
    ),
    ]
