from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.logger import Logger

logger = Logger.get_logger()

class Evaluation:
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate the model and log evaluation metrics.

        Args:
            model: Trained model to evaluate.
            X_test: Test feature set.
            y_test: Test target values.
        """
        try:
            predictions = model.predict(X_test)

            # Confusion Matrix
            cm = confusion_matrix(y_test, predictions)
            logger.info("Confusion Matrix:")
            logger.info(f"\n{cm}")

            # Classification Report
            cr = classification_report(y_test, predictions)
            logger.info("Classification Report:")
            logger.info(f"\n{cr}")

            # Accuracy Score
            accuracy = accuracy_score(y_test, predictions)
            logger.info(f"Accuracy Score: {accuracy:.2f}")

            print("Evaluation completed successfully. Check logs for details.")
        except Exception as e:
            logger.exception("An error occurred during model evaluation.")
            raise

    @staticmethod
    def compute_accuracy(predictions, y_test):
        """
        Compute and return the accuracy score.

        Args:
            predictions: Model predictions.
            y_test: True target values.

        Returns:
            Accuracy score as a float.
        """
        try:
            return accuracy_score(y_test, predictions)
        except Exception as e:
            logger.exception("An error occurred while computing accuracy.")
            raise
