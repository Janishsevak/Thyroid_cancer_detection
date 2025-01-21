import pandas as pd
from src.logger import Logger

logger = Logger.get_logger()

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load the dataset from the file."""
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.exception("An error occurred while loading the dataset.")
            raise

    def preprocess_data(self, data):
        """Preprocess the dataset."""
        try:
            # Convert 'Recurred' target variable to binary (0: No, 1: Yes)
            data['Recurred'] = data['Recurred'].map({'No': 0, 'Yes': 1})
            
            # One-hot encoding for categorical columns
            categorical_columns = [
                'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 
                'Thyroid Function', 'Physical Examination', 'Adenopathy', 
                'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
            ]
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

            logger.info(f"Preprocessing completed. Data shape after preprocessing: {data.shape}")

            # Separate features and target variable
            X = data.drop('Recurred', axis=1)
            y = data['Recurred']
            return X, y
        except KeyError as e:
            logger.error(f"Key error during preprocessing: {e}")
            raise
        except Exception as e:
            logger.exception("An error occurred during data preprocessing.")
            raise
