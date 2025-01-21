from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FeatureEngineering:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def split_data(self, X, y):
        """Split the dataset into training and testing sets."""
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale_features(self, X_train, X_test):
        """Scale the features using StandardScaler."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
