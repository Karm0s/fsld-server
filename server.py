import os
from socket import socket
from cv2 import exp

from flask import Flask
from flask_socketio import SocketIO

import numpy as np

from server.dl_models import DLModel

app = Flask(__name__)
socketio = SocketIO(app=app, cors_allowed_origins='*')

@socketio.on('connect')
def test_connect():
    print("[!] Connection established with client.")
    socketio.emit('after connect', {"connected":True})

@socketio.on('mediapipe-data')
def receive_mediapipe_data(data):
    print('[!] received array of data of length:')
    new_data = np.expand_dims(np.asarray(data), axis=0)
    print(new_data.shape)
    res = model.predict(new_data)
    word = model.get_words_list()[np.argmax(res)] 
    print(word)
    socketio.emit('prediction', word)

if __name__=="__main__":
    print('** Loading Model')
    model = DLModel()
    print('** Loading Model')
    model.load_weights('new_weights.h5')
    socketio.run(app)
