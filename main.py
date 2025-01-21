import os
import mlflow
import mlflow.sklearn
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.model import Model
from src.evaluation import Evaluation
from src.logger import Logger
import dagshub
dagshub.init(repo_owner='Janishsevak', repo_name='Thyroid_cancer_detection', mlflow=True)

logger = Logger.get_logger()


# Paths
DATA_PATH = 'data/dataset.csv'
MODEL_PATH = 'artifacts/model.pkl'

def main():
    try:
        # Start an MLflow experiment
        mlflow.set_experiment("Thyroid Cancer Detection")
        with mlflow.start_run():
            # Step 1: Load and preprocess data
            loader = DataLoader(DATA_PATH)
            data = loader.load_data()
            X, y = loader.preprocess_data(data)

            # Log dataset information
            mlflow.log_param("Dataset Size", X.shape[0])
            mlflow.log_param("Number of Features", X.shape[1])

            # Step 2: Feature engineering
            fe = FeatureEngineering()
            X_train, X_test, y_train, y_test = fe.split_data(X, y)
            X_train_scaled, X_test_scaled = fe.scale_features(X_train, X_test)

            # Step 3: Train model
            model_obj = Model()
            model_obj.train(X_train_scaled, y_train)
            model_obj.save_model(MODEL_PATH)

            # Log the model to DagsHub
            mlflow.sklearn.log_model(model_obj.model, "model")
            logger.info(f"Model saved and logged to DagsHub via MLflow.")

            # Step 4: Evaluate model
            Evaluation.evaluate_model(model_obj.model, X_test_scaled, y_test)

            # Compute and log metrics
            predictions = model_obj.model.predict(X_test_scaled)
            accuracy = Evaluation.compute_accuracy(predictions, y_test)
            logger.info(f"Computed Accuracy: {accuracy:.2f}")
            mlflow.log_metric("Accuracy", accuracy)

            logger.info(f"Accuracy logged to DagsHub: {accuracy:.2f}")

    except Exception as e:
        logger.exception("An error occurred in the main workflow.")
        raise

if __name__ == "__main__":
    main()
