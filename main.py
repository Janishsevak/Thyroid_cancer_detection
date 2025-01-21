import os
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.model import Model
from src.evaluation import Evaluation
from src.logger import Logger

logger = Logger.get_logger()

def main():
    # Paths
    DATA_PATH = 'Data/dataset.csv'  
    MODEL_PATH = 'artifacts/model.pkl'

    try:
        # Load and preprocess data
        loader = DataLoader(DATA_PATH)
        data = loader.load_data()
        X, y = loader.preprocess_data(data)

        # Feature engineering
        fe = FeatureEngineering()
        X_train, X_test, y_train, y_test = fe.split_data(X, y)
        X_train_scaled, X_test_scaled = fe.scale_features(X_train, X_test)

        # Train and save the model
        model_obj = Model()
        model_obj.train(X_train_scaled, y_train)
        model_obj.save_model(MODEL_PATH)

        # Evaluate the model
        Evaluation.evaluate_model(model_obj.model, X_test_scaled, y_test)

    except Exception as e:
        logger.exception("An error occurred in the main workflow.")
        raise

if __name__ == "__main__":
    main()
