
from data.load import DataLoader
from features.preprocess import Preprocessor
from models.train import ModelTrainer
from features import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def main():
    print("\n Step 1: Loading data...")
    dataload = DataLoader(train_path="data/train.csv", test_path="data/test.csv")

    print("\n Step 2: Preprocessing data...")

    preprocess = Preprocessor(dataload)
    train_data, categorical_features, numerical_features = preprocess.preprocess()
    selected_features = numerical_features + categorical_features

    model_trainer = ModelTrainer(train_data, numerical_features, categorical_features)

    

    X_train, X_test, y_train, y_test = model_trainer.prepare_data()




    # X = train_data[selected_features]
    # y = train_data['SalePrice']
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    print("\n Step 3: Training model...")

    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features),
    ]
    )
    rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

    model = Pipeline([
    ("preprocess", preprocessor),
    ("model", rf)
    ])

    model.fit(X_train, y_train)
    print("\n Step 4: Evaluating model...")
    y_pred = model.predict(X_test)
    print("\n Step 5: Metrics...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n Mean Squared Error: {round(mse, 3)}")
    print(f" R-squared Score: {round(r2, 3)}")



if __name__ == "__main__":
    main()
