import datetime
from dash import Input, Output
from dash import dcc, html
import dash_bootstrap_components as dbc


class Demo:
    def __init__(self, app):
        self.app = app

        @app.callback(
            Output(component_id='count_time', component_property='children'),
            Input(component_id='interval-component',
                  component_property='n_intervals')
        )
        def update_output_div(input_value):
            now = datetime.datetime.now()
            return html.Div(
                now.strftime("%H:%M:%S"),
                style={'fontSize': '50px', 'color': 'red'}
            )
