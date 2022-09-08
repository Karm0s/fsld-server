from flask import Flask
from flask_socketio import SocketIO

import numpy as np

from dl_models import DLModel

app = Flask(__name__)
socketio = SocketIO(app=app, cors_allowed_origins='*')


def construct_response_object(words, probabilities):
    response = []
    print(words)
    print(probabilities)
    for (i, word) in enumerate(words):
        response.append(
            {'word': word, 'probability': float(probabilities[0][i])})
    print(response)
    return response


@socketio.on('connect')
def test_connect():
    print("[!] Connection established with client.")
    socketio.emit('after connect', {"connected": True})


@socketio.on('mediapipe-data')
def receive_mediapipe_data(data):
    print('[!] received array of data of length:')
    new_data = np.expand_dims(np.asarray(data), axis=0)
    print(new_data.shape)
    res = model.predict(new_data)
    response = construct_response_object(model.get_words_list(), res)
    socketio.emit('predictions', response)


if __name__ == "__main__":
    print('** Loading Model')
    model = DLModel()
    print('** Loading Model')
    model.load_weights('new_weights.h5')
    socketio.run(app)
