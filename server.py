from flask import Flask
from flask_socketio import SocketIO

import numpy as np

from dl_models import DLModel

app = Flask(__name__)
socketio = SocketIO(app=app, cors_allowed_origins='*')
model = DLModel()
model.load_weights('new_weights.h5')


def construct_response_object(words, np_probabilities):
    probabilities = []
    for (i, word) in enumerate(words):
        probabilities.append(
            {'word': word, 'probability': float(np_probabilities[0][i])})

    max_probability = max(probabilities, key=lambda x: x['probability'])

    return {
        'maxProbability': max_probability,
        'probabilities': probabilities
    }


@socketio.on('connect')
def test_connect():
    print("[!] Connection established with client.")
    socketio.emit('after connect', {"connected": True})


@socketio.on('mediapipe-data')
def receive_mediapipe_data(data):
    new_data = np.expand_dims(np.asarray(data), axis=0)
    res = model.predict(new_data)
    response = construct_response_object(model.get_words_list(), res)
    socketio.emit('predictions', response)
