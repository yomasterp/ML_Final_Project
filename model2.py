from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def buildLinearModel(data):
    """Build and evaluate the Linear Regression model."""
    # Define features and target variable
    numeric_features = ['Smoothed Kills', 'Weighted Kills Against Opposing Team', 'Normalized KAST',
                        'ACS per Death', 'KDR', 'Impact Score', 'KAD Ratio']

    X = data[numeric_features]
    y = data['Kills']

    # Create a preprocessing pipeline with imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Apply the preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

    # Preprocess the features
    X_processed = preprocessor.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Initialize the Linear Regression model
    lr_model = LinearRegression()

    # Train the model
    lr_model.fit(X_train, y_train)

    # Make predictions
    predictions = lr_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Linear Regression Model Evaluation: \n Predictions: {predictions.mean(): .4f} \nMean Squared Error: {mse:.4f}\nR² Score: {r2:.4f}")

    return predictions, mse, r2
