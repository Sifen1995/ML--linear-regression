import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os


TRAIN_CSV_FILE = 'train.csv'
TEST_CSV_FILE = 'test.csv'
SAMPLE_SUBMISSION_CSV_FILE = 'sample_submission.csv'
OUTPUT_SUBMISSION_FILE = 'submission.csv' 


SQUARE_FOOTAGE_COL = 'GrLivArea' 
BEDROOMS_COL = 'BedroomAbvGr'  
BATHROOMS_COL = 'FullBath'     
PRICE_COL = 'SalePrice'        


print(f"--- Loading Training Data from {TRAIN_CSV_FILE} ---")
try:
    script_dir = os.path.dirname(__file__)
    train_file_path = os.path.join(script_dir, TRAIN_CSV_FILE)
    train_df = pd.read_csv(train_file_path)
    print("Training data loaded successfully!")
    print(f"Shape of training data: {train_df.shape}")
except FileNotFoundError:
    print(f"Error: The training file '{TRAIN_CSV_FILE}' was not found in '{script_dir}'.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the training CSV file: {e}")
    exit()


print("\n--- Exploratory Data Analysis on Training Data ---")
print("\nDataFrame Head (Train):")
print(train_df.head())

print("\nDataFrame Info (Train):")
train_df.info()

print("\nDataFrame Description (Train):")
print(train_df.describe())

print("\nMissing Values Check (Train):")
print(train_df.isnull().sum())


required_cols = [SQUARE_FOOTAGE_COL, BEDROOMS_COL, BATHROOMS_COL, PRICE_COL]
if all(col in train_df.columns for col in required_cols):
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=SQUARE_FOOTAGE_COL, y=PRICE_COL, data=train_df)
    plt.title(f'{PRICE_COL} vs. {SQUARE_FOOTAGE_COL}')

    plt.subplot(1, 3, 2)
    sns.scatterplot(x=BEDROOMS_COL, y=PRICE_COL, data=train_df)
    plt.title(f'{PRICE_COL} vs. {BEDROOMS_COL}')

    plt.subplot(1, 3, 3)
    sns.scatterplot(x=BATHROOMS_COL, y=PRICE_COL, data=train_df)
    plt.title(f'{PRICE_COL} vs. {BATHROOMS_COL}')
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_df[[SQUARE_FOOTAGE_COL, BEDROOMS_COL, BATHROOMS_COL, PRICE_COL]].corr(),
                annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features and Price (Train Data)')
    plt.show()
else:
    print("\nWarning: Not all specified feature/price columns found in train.csv for EDA. Skipping some plots.")



print("\n--- Preprocessing Training Data ---")

initial_train_rows = train_df.shape[0]
train_df.dropna(subset=[SQUARE_FOOTAGE_COL, BEDROOMS_COL, BATHROOMS_COL, PRICE_COL], inplace=True)
if train_df.shape[0] < initial_train_rows:
    print(f"Dropped {initial_train_rows - train_df.shape[0]} rows with missing essential values in training data.")
print(f"Remaining rows in training data: {train_df.shape[0]}")


features = [SQUARE_FOOTAGE_COL, BEDROOMS_COL, BATHROOMS_COL]
X = train_df[features]
y = train_df[PRICE_COL]


if not pd.api.types.is_numeric_dtype(y):
    print(f"Warning: The '{PRICE_COL}' column in train.csv contains non-numeric values. Attempting to convert.")
    y = pd.to_numeric(y, errors='coerce') 
    train_df.dropna(subset=[PRICE_COL], inplace=True) 
    X = train_df[features] 
    y = train_df[PRICE_COL]
    if y.shape[0] < initial_train_rows:
        print(f"Dropped {initial_train_rows - y.shape[0]} rows where '{PRICE_COL}' could not be converted to numeric.")


print("\n--- Splitting Training Data for Internal Validation ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Internal Training set size (X_train): {X_train.shape[0]} samples")
print(f"Internal Testing set size (X_test): {X_test.shape[0]} samples")

# --- 5. Model Training ---
print("\n--- Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training complete.")

# Print the model coefficients and intercept
print("\n--- Model Coefficients and Intercept ---")
print(f"Intercept: {model.intercept_:.2f}")
coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("Coefficients per Feature:")
print(coefficients_df)

# --- 6. Model Evaluation (on internal test set) ---
print("\n--- Evaluating Model Performance on Internal Test Set ---")
y_pred_internal = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_internal)
mse = mean_squared_error(y_test, y_pred_internal)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_internal)

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Mean Squared Error (MSE): ${mse:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# Visualize internal predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_internal, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2) # Line of perfect prediction
plt.xlabel("Actual Prices (Internal Test Set)")
plt.ylabel("Predicted Prices (Internal Test Set)")
plt.title("Actual vs. Predicted House Prices (Internal Validation)")
plt.grid(True)
plt.show()

# --- 7. Loading and Preprocessing Test Data for Final Predictions ---
print(f"\n--- Loading Test Data from {TEST_CSV_FILE} for Final Predictions ---")
try:
    test_file_path = os.path.join(script_dir, TEST_CSV_FILE)
    test_df = pd.read_csv(test_file_path)
    print("Test data loaded successfully!")
    print(f"Shape of test data: {test_df.shape}")
except FileNotFoundError:
    print(f"Error: The test file '{TEST_CSV_FILE}' was not found in '{script_dir}'.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the test CSV file: {e}")
    exit()


initial_test_rows = test_df.shape[0]
test_df.dropna(subset=features, inplace=True)
if test_df.shape[0] < initial_test_rows:
    print(f"Dropped {initial_test_rows - test_df.shape[0]} rows with missing essential values in test data.")
print(f"Remaining rows in test data: {test_df.shape[0]}")

X_final_test = test_df[features]

# --- 8. Making Final Predictions on the `test.csv` Data ---
print("\n--- Making Final Predictions on the Test Data ---")
final_predictions = model.predict(X_final_test)

# --- 9. Creating the Submission File ---
print(f"\n--- Creating Submission File: {OUTPUT_SUBMISSION_FILE} ---")

try:
    sample_submission_file_path = os.path.join(script_dir, SAMPLE_SUBMISSION_CSV_FILE)
    sample_submission_df = pd.read_csv(sample_submission_file_path)
    print(f"Sample submission loaded successfully from {SAMPLE_SUBMISSION_CSV_FILE}")
except FileNotFoundError:
    print(f"Warning: Sample submission file '{SAMPLE_SUBMISSION_CSV_FILE}' not found.")
    print("Assuming submission file needs 'Id' column from test_df and a 'Price' column for predictions.")
    sample_submission_df = pd.DataFrame({'Id': test_df['Id']}) # Assuming test_df has an 'Id' column


submission_df = pd.DataFrame({
    'Id': test_df['Id'], 
    PRICE_COL: final_predictions 
})

# Save the submission file
submission_output_path = os.path.join(script_dir, OUTPUT_SUBMISSION_FILE)
submission_df.to_csv(submission_output_path, index=False) 

print(f"Submission file '{OUTPUT_SUBMISSION_FILE}' created successfully at {os.path.abspath(submission_output_path)}")
print("You can now submit this file to the competition!")