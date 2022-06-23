from dash import Input, Output


class Demo:
    def __init__(self, app):
        self.app = app

        @app.callback(
            Output(component_id='my-output', component_property='children'),
            Input(component_id='my-input', component_property='value')
        )
        def update_output_div(input_value):
            return f'Output: {input_value}'
