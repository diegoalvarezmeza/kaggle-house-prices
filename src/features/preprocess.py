import pandas as pd

from data.load import DataLoader

class Preprocessor:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def load(self):
        """Preprocess the loaded data."""
        train_data, test_data = self.data_loader.load_data()
   
        return train_data, test_data
    
    def split_numerical_categoric(self, df):
        """Split the dataframe into numerical and categorical features."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categoric_cols = df.select_dtypes(include=['object']).columns.tolist()
        return numerical_cols, categoric_cols
    

    def drop_columns_na(self, df, threshold=0.5):
        """Drop columns with more than threshold percentage of missing values."""
        thresh_count = int(threshold * len(df))
        df_dropped = df.dropna(axis=1, thresh=thresh_count)
        return df_dropped
    
    def select_numerical_features(self, df, n_best_correlation):
        """Select top n numerical features based on correlation with target."""
        numerical_cols, _ = self.split_numerical_categoric(df)
        corr_matrix = df[numerical_cols].corr()
        target_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False)
        top_features = target_corr.index[1:n_best_correlation+1].tolist()  # Exclude target itself
        return top_features
    
    def preprocess(self, n_best_correlation=10, na_threshold=0.5):
        """Full preprocessing pipeline."""
        train_data, _ = self.load()
        
        # Drop columns with too many missing values
        train_data = self.drop_columns_na(train_data, threshold=na_threshold)

        numerical_cols, categoric_cols = self.split_numerical_categoric(train_data)
    
        # Select top numerical features
        top_numerical_features = self.select_numerical_features(train_data, n_best_correlation)

        return numerical_cols, top_numerical_features
    
        

        