import pickle
import pandas as pd

class Predictor:
    @staticmethod
    def load_model(file_path):
        """Load the trained model."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def predict(model, data):
        """Make predictions on new data."""
        predictions = model.predict(data)
        return predictions
