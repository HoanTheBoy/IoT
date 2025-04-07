import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings

# Tùy chọn: Bỏ qua các cảnh báo từ Optuna hoặc các thư viện khác (nếu cần)
# warnings.filterwarnings('ignore')
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. Tải Dữ Liệu ---
print("--- 1. Loading Data ---")
try:
    df = pd.read_csv('dataset2.csv')
    print("Dataset loaded successfully.")
    print("Initial shape:", df.shape)
    # print(df.head())
    # print(df.info())
except FileNotFoundError:
    print("Error: dataset2.csv not found. Make sure the file is in the correct directory.")
    exit()

# --- 2. Làm Sạch Cơ Bản ---
print("\n--- 2. Basic Cleaning ---")
# Loại bỏ cột 'Name' vì nó là mô tả text và thông tin có thể đã có ở các cột khác
if 'Name' in df.columns:
    df = df.drop('Name', axis=1)
    print("Dropped 'Name' column.")
else:
    print("'Name' column not found.")

# Kiểm tra giá trị thiếu (ví dụ đơn giản)
# print("Missing values before handling:\n", df.isnull().sum())
# Xử lý giá trị thiếu nếu có (ví dụ: xóa hàng hoặc điền giá trị)
# df.dropna(inplace=True) # Ví dụ: xóa hàng có giá trị thiếu

print("Shape after basic cleaning:", df.shape)


# --- 3. Xác Định Cột và Tạo Pipeline Tiền Xử Lý ---
print("\n--- 3. Defining Features and Preprocessing Pipeline ---")
# Xác định các loại cột
numerical_cols = ['Storage', 'ScreenSize', 'RAM', 'Battery_Cells', 'Battery_Wh']
categorical_cols = ['CPU_Manufacturer', 'CPU_Series', 'GPU_Type', 'Shell']
target_col = 'Price'

# Kiểm tra xem tất cả các cột đã xác định có tồn tại trong DataFrame không
all_cols = numerical_cols + categorical_cols + [target_col]
missing_cols = [col for col in all_cols if col not in df.columns]
if missing_cols:
    print(f"Error: The following specified columns are missing from the DataFrame: {missing_cols}")
    exit()
extra_cols = [col for col in df.columns if col not in all_cols]
if extra_cols:
    print(f"Warning: The following columns exist in the DataFrame but were not specified as features or target: {extra_cols}")
    print("These columns will be dropped.")
    df = df[all_cols] # Giữ lại các cột đã định nghĩa

# Tạo pipeline cho biến số
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Tạo pipeline cho biến phân loại
categorical_transformer = Pipeline(steps=[
    # handle_unknown='ignore': Bỏ qua các giá trị không thấy trong lúc fit
    # drop='first': Tránh bẫy biến giả (dummy variable trap)
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])

# Kết hợp các pipeline bằng ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop' # Bỏ qua các cột không được chỉ định trong transformers
)

# --- 4. Phân Chia Dữ Liệu và Xử Lý Biến Mục Tiêu ---
print("\n--- 4. Splitting Data and Handling Target Variable ---")
X = df.drop(target_col, axis=1)
y = df[target_col]

# Áp dụng log transform cho giá để giảm độ lệch (phổ biến cho biến tiền tệ)
y_log = np.log1p(y)
print("Applied log1p transformation to the target variable 'Price'.")

# Phân chia dữ liệu (80% train, 20% test)
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

print(f"Training set shape: X_train={X_train.shape}, y_train_log={y_train_log.shape}")
print(f"Testing set shape: X_test={X_test.shape}, y_test_log={y_test_log.shape}")

# --- 5. Áp Dụng Tiền Xử Lý ---
print("\n--- 5. Applying Preprocessing ---")
# Fit preprocessor trên tập train và transform cả train/test
# Quan trọng: chỉ fit trên tập train để tránh rò rỉ dữ liệu từ tập test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Lấy tên các cột sau khi biến đổi (hữu ích cho phân tích sau này)
try:
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"Number of features after preprocessing: {len(feature_names_out)}")
    # print("Feature names after preprocessing:", feature_names_out) # Có thể rất dài
except Exception as e:
    print(f"Warning: Could not get feature names after preprocessing. Error: {e}")
    feature_names_out = None

print("Shape of X_train_processed:", X_train_processed.shape)
print("Shape of X_test_processed:", X_test_processed.shape)


# --- 6. Huấn Luyện và Đánh Giá Các Mô Hình Cơ Bản ---
print("\n--- 6. Training and Evaluating Base Models ---")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_processed, y_train_log)
    y_pred_log = model.predict(X_test_processed)

    # Inverse transform để có giá dự đoán thực tế
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test_log) # Inverse transform y_test cũng để so sánh trên thang đo gốc

    # Tính toán các chỉ số lỗi trên thang đo giá gốc
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual) # R2 có thể tính trên log hoặc thang gốc

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"--- {name} Evaluation (Original Price Scale) ---")
    print(f"MAE: {mae:,.0f}") # Format tiền tệ không có số thập phân
    print(f"RMSE: {rmse:,.0f}")
    print(f"R2: {r2:.4f}\n")

# In bảng kết quả
results_df = pd.DataFrame(results).T.sort_values(by="RMSE")
print("--- Model Comparison (Sorted by RMSE) ---")
print(results_df)


# --- 7. Phân Tích Độ Quan Trọng Đặc Trưng ---
print("\n--- 7. Feature Importance Analysis (Using Random Forest) ---")
# Sử dụng mô hình Random Forest đã huấn luyện ở trên
rf_model = models["Random Forest"]

if hasattr(rf_model, 'feature_importances_') and feature_names_out is not None:
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Top 15 Feature Importances:")
    print(feature_importance_df.head(15))

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.show() # Bỏ comment nếu muốn hiển thị plot ngay lập tức
    # plt.savefig('feature_importance.png') # Lưu plot ra file
    # print("Feature importance plot saved as feature_importance.png")

else:
    print("Could not perform feature importance analysis (model doesn't support it or feature names unavailable).")


# --- 8. Tối Ưu Hóa Siêu Tham Số (Optuna for Random Forest) ---
print("\n--- 8. Hyperparameter Optimization with Optuna (Random Forest) ---")

# Hàm mục tiêu để Optuna tối ưu hóa
def objective(trial):
    # Định nghĩa không gian tìm kiếm siêu tham số
    n_estimators = trial.suggest_int('n_estimators', 50, 400, step=50)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_float('max_features', 0.1, 1.0) # Tỷ lệ features sử dụng

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Đánh giá bằng cross-validation trên tập train (thang đo log)
    # Sử dụng neg_root_mean_squared_error để tối thiểu hóa RMSE
    score = cross_val_score(
        model, X_train_processed, y_train_log, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
    )
    rmse_cv = -score.mean() # Lấy giá trị dương của RMSE
    return rmse_cv

# Tạo và chạy study của Optuna
# sampler = optuna.samplers.TPESampler(seed=42) # Để kết quả có thể tái lặp
study = optuna.create_study(direction='minimize') #, sampler=sampler)
study.optimize(objective, n_trials=50, timeout=600) # Giới hạn số lần thử hoặc thời gian

print("\nOptuna Optimization Finished.")
print("Best trial:")
best_trial = study.best_trial
print(f"  Value (RMSE on Log Scale CV): {best_trial.value:.4f}")
print("  Best Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# --- 9. Huấn Luyện và Đánh Giá Mô Hình Cuối Cùng (Đã Tối Ưu) ---
print("\n--- 9. Training and Evaluating Final Tuned Model ---")
best_params = best_trial.params
final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

print("Training final model with best parameters...")
final_model.fit(X_train_processed, y_train_log)

print("Evaluating final model on the test set...")
y_pred_final_log = final_model.predict(X_test_processed)

# Inverse transform để có giá gốc
y_pred_final_actual = np.expm1(y_pred_final_log)
y_test_actual = np.expm1(y_test_log) # Đã tính ở trên nhưng gọi lại cho rõ ràng

# Tính toán chỉ số cuối cùng
final_mae = mean_absolute_error(y_test_actual, y_pred_final_actual)
final_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_final_actual))
final_r2 = r2_score(y_test_actual, y_pred_final_actual)

print("\n--- Final Tuned Model Evaluation (Original Price Scale) ---")
print(f"MAE: {final_mae:,.0f}")
print(f"RMSE: {final_rmse:,.0f}")
print(f"R2: {final_r2:.4f}")

# (Tùy chọn) So sánh kết quả mô hình cuối cùng với mô hình RF cơ bản
print("\nComparison with base Random Forest model:")
print(f"Base RF   -> MAE: {results['Random Forest']['MAE']:,.0f}, RMSE: {results['Random Forest']['RMSE']:,.0f}, R2: {results['Random Forest']['R2']:.4f}")
print(f"Tuned RF  -> MAE: {final_mae:,.0f}, RMSE: {final_rmse:,.0f}, R2: {final_r2:.4f}")


# --- Kết thúc ---
print("\n--- Script Finished ---")