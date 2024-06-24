import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('car_data.csv')  # Ensure your CSV file is named 'car_data.csv'

# Check the first few rows of the dataset to understand its structure
print(data.head())

# Separate features and target variable
X = data.drop(['Selling_Price'], axis=1)  # Drop the target column 'Selling_Price'
y = data['Selling_Price']  # Target variable

# Identify categorical and numerical columns
categorical_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']  # Categorical columns
numerical_cols = ['Year', 'Present_Price', 'Driven_kms', 'Owner']  # Numerical columns

# Preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes the preprocessor and the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Save the trained model to a file
joblib.dump(pipeline, 'car_price_prediction_model.pkl')
print("Model saved as car_price_prediction_model.pkl")
