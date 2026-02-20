import pandas as pd
from data.load import DataLoader
from features.preprocess import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, df, numerical_features, categorical_features):
        self.df = df
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_data(self):
        """Load and preprocess the data."""
        # train_data, test_data = self.dataloader.load_data()
        # self.preprocessor = Preprocessor(self.dataloader)
        # numerical_features, categorical_features = self.preprocessor.preprocess()
        selected_features = self.numerical_features + self.categorical_features

        X = self.df[selected_features]
        y = self.df['SalePrice']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    

    # def prepare_data(self):
    #     """Load and preprocess the data."""
    #     train_data, test_data = self.dataloader.load_data()
    #     self.preprocessor = Preprocessor(self.dataloader)
    #     numerical_features, categorical_features = self.preprocessor.preprocess()
    #     selected_features = numerical_features + categorical_features

    #     X = train_data[selected_features]
    #     y = train_data['SalePrice']

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     return X_train, X_test, y_train, y_test

    def train_model(self):
        """Train the Random Forest model."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.model.fit(X_train, y_train)
        return self.model


class RegressionModel:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)