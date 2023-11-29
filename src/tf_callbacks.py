from . import utils
from tensorflow import keras


class Callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, feature_names, target, riskiest_features):
        self.training_data = training_data
        self.validation_data = validation_data
        self.feature_names = feature_names
        self.target = target
        self.riskiest_features = riskiest_features

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 10 != 0:
            return
        print()
        print(f"Epoch {epoch+1}")
        print("Training Metrics")
        utils.corr(self.training_data, self.model, self.feature_names, self.target, self.riskiest_features)
        print()
        print("Validation Metrics")
        utils.corr(self.validation_data, self.model, self.feature_names, self.target, self.riskiest_features)
        print()
        print()