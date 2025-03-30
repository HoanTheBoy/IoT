# --- START OF PYTHON CODE ---

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re # Using standard re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import warnings as w
w.filterwarnings('ignore') # Suppress warnings for cleaner output
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib as j

# --- 1. Data Loading and Initial Exploration ---
print("--- Loading Data ---")
# Read data from laptops.csv
try:
    df = pd.read_csv('laptop.csv')
    print("Data loaded successfully.")
    print("Original DataFrame shape:", df.shape)
except FileNotFoundError:
    print("Error: laptops.csv not found. Please make sure the file is in the correct directory.")
    exit() # Exit if file not found

# --- 2. Data Cleaning and Preprocessing ---
print("\n--- Data Cleaning and Preprocessing ---")

# Display basic info
print("\nDataFrame Info (Initial):")
df.info()

# Check for missing values
print("\nMissing Values (Initial):")
print(df.isna().sum())

# Handle missing values
# GPU: Fill missing GPU values with 'No GPU' or a common integrated GPU like 'Intel HD Graphics' / 'Intel UHD Graphics' or mode
# For simplicity, let's fill with 'Unknown'
df['GPU'].fillna('Unknown', inplace=True)
# Storage type: Seems mostly SSD or eMMC. Let's check if NaN exists and fill if needed.
# The provided CSV has empty strings in GPU, not NaN. Let's replace empty strings too.
df['GPU'] = df['GPU'].replace('', 'Unknown')
# Storage type might have missing values if Storage is 0 (like in one refurbished Alurin).
# Fill missing Storage type based on common types or 'Unknown'
df['Storage type'].fillna('Unknown', inplace=True)
df['Storage type'] = df['Storage type'].replace('', 'Unknown')

# Clean 'Laptop' column - it's too specific for general modeling, let's drop it
# We already have Brand, Model, CPU, RAM etc.
df.drop('Laptop', axis=1, inplace=True)
print("\n'Laptop' column dropped.")

# Convert RAM to numeric (already numeric in the provided CSV)
# df['RAM'] = df['RAM'].astype(int) # Already integer

# Convert Storage to numeric (already numeric in the provided CSV)
# df['Storage'] = df['Storage'].astype(int) # Already integer

# Convert Screen to numeric (already numeric in the provided CSV)
# df['Screen'] = df['Screen'].astype(float) # Already float

# Convert Touch to numeric (Yes/No -> 1/0)
df['Touch'] = df['Touch'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)
print("\n'Touch' column converted to numeric (1/0).")

# Clean Final Price (remove potential commas if any, already float)
# df['Final Price'] = df['Final Price'].astype(float) # Already float

# --- 3. Feature Engineering ---
print("\n--- Feature Engineering ---")

# Extract CPU Brand
def extract_cpu_brand(cpu_name):
    cpu_name = str(cpu_name).lower() # Ensure string and lower case
    if 'intel' in cpu_name:
        if 'i9' in cpu_name: return 'Intel Core i9'
        if 'i7' in cpu_name: return 'Intel Core i7'
        if 'i5' in cpu_name: return 'Intel Core i5'
        if 'i3' in cpu_name: return 'Intel Core i3'
        if 'pentium' in cpu_name: return 'Intel Pentium'
        if 'celeron' in cpu_name: return 'Intel Celeron'
        if 'atom' in cpu_name: return 'Intel Atom'
        return 'Other Intel' # For Xeon, Core M etc.
    elif 'amd' in cpu_name:
        if 'ryzen 9' in cpu_name: return 'AMD Ryzen 9'
        if 'ryzen 7' in cpu_name: return 'AMD Ryzen 7'
        if 'ryzen 5' in cpu_name: return 'AMD Ryzen 5'
        if 'ryzen 3' in cpu_name: return 'AMD Ryzen 3'
        if 'athlon' in cpu_name: return 'AMD Athlon'
        return 'Other AMD' # For A-series, E-series etc.
    elif 'apple' in cpu_name or 'm1' in cpu_name or 'm2' in cpu_name:
        return 'Apple Silicon'
    elif 'qualcomm' in cpu_name or 'snapdragon' in cpu_name:
        return 'Qualcomm Snapdragon'
    elif 'mediatek' in cpu_name:
        return 'MediaTek'
    else:
        return 'Other'

df['CPU_Brand'] = df['CPU'].apply(extract_cpu_brand)
print("\n'CPU_Brand' extracted.")
# df.drop('CPU', axis=1, inplace=True) # Keep original CPU for now if needed for more detail later

# Extract GPU Brand
def extract_gpu_brand(gpu_name):
    gpu_name = str(gpu_name).lower() # Ensure string and lower case
    if 'nvidia' in gpu_name or 'rtx' in gpu_name or 'gtx' in gpu_name or 'geforce' in gpu_name or 'quadro' in gpu_name or 'mx' in gpu_name:
        return 'Nvidia'
    elif 'amd' in gpu_name or 'radeon' in gpu_name or 'firepro' in gpu_name:
        return 'AMD'
    elif 'intel' in gpu_name or 'iris' in gpu_name or 'uhd' in gpu_name or 'hd graphics' in gpu_name:
        return 'Intel'
    elif 'apple' in gpu_name:
         return 'Apple'
    elif 'qualcomm' in gpu_name or 'adreno' in gpu_name:
        return 'Qualcomm'
    elif 'arm' in gpu_name or 'mali' in gpu_name:
        return 'ARM'
    elif gpu_name == 'unknown' or gpu_name == '':
        return 'Unknown' # Explicitly handle unknowns
    else:
        return 'Other' # Catch-all for less common ones

df['GPU_Brand'] = df['GPU'].apply(extract_gpu_brand)
print("\n'GPU_Brand' extracted.")
# df.drop('GPU', axis=1, inplace=True) # Keep original GPU for now

# We already have Storage and Storage type, which is better than trying to parse the old 'Storage' column
# Let's keep 'Storage' (size in GB) and 'Storage type'

# Drop columns not directly used in the model (or too specific)
df.drop(['Model', 'CPU', 'GPU'], axis=1, inplace=True) # Dropping original detailed columns
print("\nDropped 'Model', 'CPU', 'GPU' columns.")

print("\nFinal DataFrame Columns for Modeling:")
print(df.columns)
print("\nSample of Processed Data:")
print(df.sample(5))

# Check data types again after processing
print("\nDataFrame Info (After Processing):")
df.info()

# --- 4. Data Splitting ---
print("\n--- Splitting Data ---")
X = df.drop('Final Price', axis=1)
# Apply log transformation to the target variable if its distribution is skewed
# Let's check the distribution first
sns.histplot(df['Final Price'], kde=True)
plt.title('Distribution of Final Price (Original)')
plt.show()
# If skewed, apply log transform
print("Applying log transformation to 'Final Price'.")
y = np.log(df['Final Price'])
sns.histplot(y, kde=True)
plt.title('Distribution of Final Price (Log Transformed)')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- 5. Preprocessing Pipeline ---
print("\n--- Building Preprocessing Pipeline ---")

# Identify categorical and numerical columns for the transformer
# Be careful with column indices if you changed the column order
categorical_features = ['Status', 'Brand', 'Storage type', 'CPU_Brand', 'GPU_Brand'] # Added Status
numerical_features = ['RAM', 'Storage', 'Screen', 'Weight', 'Touch', # Added Weight, Touch
                      # 'Ips', 'PPI' # We don't have these directly from this CSV
                     ]

# Ensure all columns exist
missing_cols = [col for col in categorical_features + numerical_features if col not in X_train.columns]
if missing_cols:
    print(f"Error: The following columns are missing from X_train: {missing_cols}")
    exit()

# Create the ColumnTransformer
# Handle unknown categories in OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_features), # handle_unknown='ignore' is important for unseen test data
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough' # Keep other columns (like Touch) if not explicitly transformed
)


# --- 6. Model Training and Evaluation ---
print("\n--- Training and Evaluating Models ---")

models = {
    "LR": LinearRegression(),
    "RFC": RandomForestRegressor(n_estimators=100, random_state=42), # Reduced estimators for faster run
    "DT": DecisionTreeRegressor(random_state=42),
    "ADR": AdaBoostRegressor(random_state=42),
    "GBR": GradientBoostingRegressor(random_state=42),
    "ETR": ExtraTreesRegressor(random_state=42),
    "XGB": XGBRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    # Evaluate the model
    test_r2 = r2_score(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    mae = mean_absolute_error(y_test, y_pred_test)

    results[name] = {'Test R2': test_r2, 'Train R2': train_r2, 'MAE': mae}

    print(f"{name} - Test R2: {test_r2:.4f}, Train R2: {train_r2:.4f}, MAE: {mae:.4f}")

# --- 7. Results Comparison ---
print("\n--- Model Comparison ---")
score_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
score_df = score_df.sort_values(by='Test R2', ascending=False)
print(score_df)

# Find the best model based on Test R2
best_model_name = score_df.iloc[0]['Model']
print(f"\nBest Model based on Test R2: {best_model_name}")

plt.figure(figsize=(10, 6))
sns.barplot(data=score_df, x='Model', y='Test R2')
plt.title('Model Comparison - Test R2 Score')
plt.ylim(0, 1) # R2 score is between -inf and 1
plt.show()

# --- 8. Model Saving ---
print("\n--- Saving Best Model and Processed Data (Optional) ---")

# Retrieve the best pipeline
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', best_model)])
# Re-fit the best pipeline on the entire dataset (optional, but common practice)
print(f"Re-fitting {best_model_name} on the entire dataset...")
best_pipeline.fit(X, y)
print("Re-fitting complete.")


# Save the processed DataFrame (optional)
# j.dump(df, open('processed_laptops_df.jbl', 'wb'))
# print("Processed DataFrame saved to processed_laptops_df.jbl")

# Save the best pipeline
j.dump(best_pipeline, open('laptop_price_pipeline.jbl', 'wb'))
print(f"Best pipeline ({best_model_name}) saved to laptop_price_pipeline.jbl")

print("\n--- Script Finished ---")

# --- END OF PYTHON CODE ---