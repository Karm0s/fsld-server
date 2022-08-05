import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class DLModel:
    def __init__(self) -> None:
        self.words = self.get_dataset_words()
        self.label_map = self.create_labels(self.words)
        self.sequence_length = 30
        print(self.words)
        self.model = self.create_lstm_model()

    def create_lstm_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(self.sequence_length, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.words), activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        return model

    def create_labels(self, words):
        label_map = {label: num for num, label in enumerate(words)}
        return label_map
    
    def load_weights(self, file_path):
        self.model.load_weights(file_path)
    
    def predict(self, data):
        return self.model.predict(data)
    
    def get_words_list(self):
        return self.words
    
    def get_dataset_words(self):
        words = []
        with open('word_list.txt') as file:
            for line in file.readlines():
                words.extend(line.split(','))
        return words