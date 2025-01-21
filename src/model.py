from sklearn.ensemble import RandomForestClassifier
import pickle
from src.logger import Logger

logger = Logger.get_logger()

class Model:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """Train the model."""
        try:
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.exception("An error occurred during model training.")
            raise

    def save_model(self, file_path):
        """Save the trained model."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {file_path}")
        except Exception as e:
            logger.exception("An error occurred while saving the model.")
            raise
