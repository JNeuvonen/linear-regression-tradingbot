from flask import Flask, make_response, jsonify


def create_app(shared_data):
    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        trade_engine_state = shared_data.get_trade_engine_state()
        response = make_response(jsonify(trade_engine_state))
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return app


def launch_server(shared_data):
    app = create_app(shared_data)
    app.run(host='0.0.0.0', port=8080)
