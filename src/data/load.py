import pandas as pd

class DataLoader:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        """Load training and testing data from CSV files."""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        return self.train_data, self.test_data