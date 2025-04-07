# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
# ==============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cài đặt Optuna nếu chưa có: pip install optuna
try:
    import optuna
except ImportError:
    print("Thư viện Optuna chưa được cài đặt. Vui lòng cài đặt bằng lệnh: pip install optuna")
    exit()

warnings.filterwarnings('ignore') # Ẩn các cảnh báo không quan trọng

# ==============================================================================
# CHƯƠNG 3: QUY TRÌNH XÂY DỰNG MÔ HÌNH MACHINE LEARNING
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1 Thu thập và tiền xử lý dữ liệu laptop
# ------------------------------------------------------------------------------

# 3.1.1 Thu thập dữ liệu
print("--- 3.1.1 Thu thập dữ liệu ---")
try:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('dataset17_best.csv')
    print(f"Đã đọc thành công {df.shape[0]} dòng và {df.shape[1]} cột.")
    print("5 dòng dữ liệu đầu tiên:")
    print(df.head())
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dataset17.csv. Vui lòng kiểm tra lại đường dẫn.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file CSV: {e}")
    exit()

# 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA)
print("\n--- 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA) ---")

# Kiểm tra thông tin cơ bản và dữ liệu thiếu
print("\nThông tin bộ dữ liệu:")
df.info()
print("\nKiểm tra dữ liệu thiếu:")
print(df.isnull().sum())
# Nhận xét: Bộ dữ liệu có vẻ đầy đủ, không có giá trị thiếu.

# Phân tích mối quan hệ giữa các biến số và giá (Price_VND)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_features_initial = [col for col in numerical_cols if col != 'Price_VND'] # Biến số ban đầu

print(f"\nCác biến số ban đầu: {numerical_features_initial}")
print(f"Biến mục tiêu: Price_VND")

# --- Tương quan giữa các biến số ---
plt.figure(figsize=(14, 10)) # Tăng kích thước để dễ đọc hơn
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Ma trận tương quan giữa các biến số')
plt.show()

print("\nTương quan của các biến số với Price_VND:")
price_correlation = correlation_matrix['Price_VND'].sort_values(ascending=False)
print(price_correlation)

# --- Phân tích mối quan hệ giữa biến phân loại và giá ---
categorical_cols_initial = df.select_dtypes(include='object').columns.tolist()
# Loại bỏ 'Model_Name'
if 'Model_Name' in categorical_cols_initial:
    categorical_cols_initial.remove('Model_Name')

print(f"\nCác biến phân loại ban đầu cần phân tích: {categorical_cols_initial}")

# Vẽ biểu đồ boxplot
for col in categorical_cols_initial:
    if df[col].nunique() < 20: # Giới hạn số lượng categories để vẽ
        plt.figure(figsize=(12, 6))
        # Sắp xếp các box theo giá trị trung vị để dễ so sánh hơn
        order = df.groupby(col)['Price_VND'].median().sort_values().index
        sns.boxplot(x=col, y='Price_VND', data=df, palette='viridis', order=order)
        plt.title(f'Phân phối giá (Price_VND) theo {col}')
        plt.xticks(rotation=45, ha='right')
        plt.ticklabel_format(style='plain', axis='y') # Định dạng trục y
        plt.tight_layout()
        plt.show()
    else:
        print(f"Biến '{col}' có quá nhiều giá trị ({df[col].nunique()}), bỏ qua vẽ boxplot.")

# 3.1.3 Tiền xử lý dữ liệu
print("\n--- 3.1.3 Tiền xử lý dữ liệu ---")

df_processed = df.copy()

# --- Làm sạch dữ liệu text và Feature Engineering ---
# 1. CPU Generation
def clean_cpu_gen(gen):
    gen_str = str(gen).lower()
    if 'ultra' in gen_str: return 15
    if '7000' in gen_str: return 7
    if '6000' in gen_str: return 6
    if '5000' in gen_str: return 5
    try:
        match = re.search(r'\d+', gen_str)
        return int(match.group()) if match else 0
    except:
        return 0
df_processed['CPU_Generation_Cleaned'] = df_processed['CPU_Generation'].apply(clean_cpu_gen)
print(f"\nChuẩn hóa 'CPU_Generation'. Giá trị duy nhất: {sorted(df_processed['CPU_Generation_Cleaned'].unique())}")

# 2. CPU Model
def simplify_cpu_model(model):
    model = str(model).lower()
    if 'ultra 9' in model: return 'Core Ultra 9'
    if 'ultra 7' in model: return 'Core Ultra 7'
    if 'ultra 5' in model: return 'Core Ultra 5'
    if 'core i9' in model: return 'Core i9'
    if 'core i7' in model: return 'Core i7'
    if 'core i5' in model: return 'Core i5'
    if 'core i3' in model: return 'Core i3'
    if 'ryzen 9' in model: return 'Ryzen 9'
    if 'ryzen 7' in model: return 'Ryzen 7'
    if 'ryzen 5' in model: return 'Ryzen 5'
    if 'ryzen 3' in model: return 'Ryzen 3'
    return 'Other CPU'
df_processed['CPU_Model_Simplified'] = df_processed['CPU_Model'].apply(simplify_cpu_model)
print(f"Rút gọn 'CPU_Model'. Giá trị duy nhất: {df_processed['CPU_Model_Simplified'].unique()}")

# 3. GPU Classification
def classify_gpu(row):
    gpu_type = str(row['GPU_Type']).lower()
    gpu_name = str(row['GPU_Name']).lower()
    vram = row['GPU_VRAM_GB']

    if gpu_type == 'integrated':
        if 'intel' in gpu_name:
            if 'iris' in gpu_name or 'arc' in gpu_name: return 'Integrated_Intel_High'
            else: return 'Integrated_Intel_Low'
        elif 'amd' in gpu_name or 'radeon graphics' in gpu_name: return 'Integrated_AMD'
        else: return 'Integrated_Other'
    elif gpu_type == 'dedicated':
        if 'nvidia' in gpu_name or 'geforce' in gpu_name:
            if 'rtx 40' in gpu_name or 'rtx 307' in gpu_name or 'rtx 308' in gpu_name or 'rtx 309' in gpu_name: return 'Dedicated_NVIDIA_High'
            if 'rtx 3060' in gpu_name: return 'Dedicated_NVIDIA_MidHigh'
            if 'rtx 3050' in gpu_name or 'rtx 4050' in gpu_name or 'rtx 20' in gpu_name or 'gtx 16' in gpu_name : return 'Dedicated_NVIDIA_Mid'
            if 'mx' in gpu_name: return 'Dedicated_NVIDIA_Low'
            return 'Dedicated_NVIDIA_Other'
        elif 'amd' in gpu_name or 'radeon' in gpu_name:
            if 'rx 7' in gpu_name or 'rx 68' in gpu_name or 'rx 67' in gpu_name or 'rx 66' in gpu_name: return 'Dedicated_AMD_High'
            if 'rx 6500m' in gpu_name or 'rx 7600s' in gpu_name: return 'Dedicated_AMD_Mid'
            return 'Dedicated_AMD_Low'
        elif 'intel' in gpu_name and 'arc' in gpu_name: return 'Dedicated_Intel'
        else: return 'Dedicated_Other'
    else: return 'Unknown'
df_processed['GPU_Class'] = df_processed.apply(classify_gpu, axis=1)
print(f"Phân loại 'GPU_Name'. Giá trị duy nhất: {df_processed['GPU_Class'].unique()}")

# 4. Screen Touch
df_processed['Screen_Touch_Binary'] = df_processed['Screen_Touch'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
print(f"Chuyển đổi 'Screen_Touch'. Giá trị duy nhất: {df_processed['Screen_Touch_Binary'].unique()}")

# --- Chọn Features và Target ---
# Loại bỏ các cột gốc không cần thiết hoặc đã được xử lý
cols_to_drop = ['Model_Name', 'CPU_Vendor', 'CPU_Model', 'CPU_Generation',
                'GPU_Type', 'GPU_Name', 'Screen_Touch']
df_final_features = df_processed.drop(columns=cols_to_drop)

# Xác định lại các cột numerical và categorical cuối cùng
numerical_final = df_final_features.select_dtypes(include=np.number).columns.tolist()
target_col = 'Price_VND'
if target_col in numerical_final:
    numerical_final.remove(target_col)
categorical_final = df_final_features.select_dtypes(include='object').columns.tolist()

print("\nCác features số cuối cùng:", numerical_final)
print("Các features phân loại cuối cùng:", categorical_final)
print(f"Biến mục tiêu: {target_col}")

# Tách features (X) và target (y)
X = df_final_features.drop(target_col, axis=1)
y = df_final_features[target_col]

# --- Định nghĩa Preprocessor ---
# Bộ xử lý cho biến số
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Bộ xử lý cho biến phân loại
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Bộ tiền xử lý kết hợp
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_final),
        ('cat', categorical_transformer, categorical_final)
    ],
    remainder='passthrough' # Giữ lại các cột không được xử lý (nếu có)
)

# --- Tách dữ liệu Train/Test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nKích thước tập huấn luyện: X={X_train.shape}, y={y_train.shape}")
print(f"Kích thước tập kiểm tra: X={X_test.shape}, y={y_test.shape}")

# --- Áp dụng Preprocessor ---
# Fit trên train, transform trên cả train và test
print("\nÁp dụng tiền xử lý (fit trên train, transform trên train/test)...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Lấy tên cột sau OneHotEncode
try:
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_final)
    all_feature_names_processed = numerical_final + list(ohe_feature_names)
    print(f"Tổng số features sau tiền xử lý: {len(all_feature_names_processed)}")
except Exception as e:
    print(f"Lỗi khi lấy tên feature sau OHE: {e}")
    # Ước tính số cột nếu không lấy được tên (ít chính xác hơn)
    n_numeric = len(numerical_final)
    # Ước tính số cột OHE
    n_ohe_cols = X_train_processed.shape[1] - n_numeric
    all_feature_names_processed = numerical_final + [f'cat_{i}' for i in range(n_ohe_cols)]
    print(f"Ước tính tổng số features sau tiền xử lý: {len(all_feature_names_processed)}")


# Chuyển đổi mảng numpy đã xử lý thành DataFrame Pandas
# Điều này hữu ích cho việc xem và ánh xạ feature importance
X_train_processed_df = pd.DataFrame(X_train_processed, columns=all_feature_names_processed, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=all_feature_names_processed, index=X_test.index)

print("Hoàn tất tiền xử lý dữ liệu.")

# ==============================================================================
# 3.2 Xây dựng mô hình Machine Learning (Hồi quy)
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.2.1 Huấn luyện và đánh giá các mô hình Machine Learning (Hồi quy)
# ------------------------------------------------------------------------------
print("\n--- 3.2.1 Huấn luyện và đánh giá các mô hình ---")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42, max_iter=2000), # Tăng max_iter cho Lasso
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'), # Chỉ định objective rõ ràng
    'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    #'SVR': SVR() # Có thể thêm nếu muốn, nhưng thường chậm hơn
}

results = {}
predictions = {} # Lưu trữ dự đoán để vẽ biểu đồ sau

for name, model in models.items():
    print(f"Đang huấn luyện mô hình: {name}...")
    start_time = time.time()
    model.fit(X_train_processed_df, y_train)
    y_pred = model.predict(X_test_processed_df)
    predictions[name] = y_pred # Lưu dự đoán
    end_time = time.time()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'MAE (VND)': mae,
        'RMSE (VND)': rmse,
        'R2 Score': r2,
        'Fit Time (s)': end_time - start_time
    }
    print(f"  MAE: {mae:,.0f} VND")
    print(f"  RMSE: {rmse:,.0f} VND")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Thời gian huấn luyện: {results[name]['Fit Time (s)']:.2f} giây")
    print("-" * 30)

# Tổng hợp kết quả
results_df = pd.DataFrame(results).T.sort_values(by='R2 Score', ascending=False)
print("\nBảng tổng hợp kết quả đánh giá các mô hình:")
# Định dạng lại các cột số lớn
results_df_display = results_df.copy()
results_df_display['MAE (VND)'] = results_df_display['MAE (VND)'].map('{:,.0f}'.format)
results_df_display['RMSE (VND)'] = results_df_display['RMSE (VND)'].map('{:,.0f}'.format)
results_df_display['R2 Score'] = results_df_display['R2 Score'].map('{:.4f}'.format)
results_df_display['Fit Time (s)'] = results_df_display['Fit Time (s)'].map('{:.2f}'.format)
print(results_df_display)

best_model_name = results_df.index[0]
print(f"\nMô hình tốt nhất dựa trên R2 Score: {best_model_name}")

# ------------------------------------------------------------------------------
# 3.2.2 Phân tích mức độ quan trọng của các đặc trưng (Thông số kỹ thuật)
# ------------------------------------------------------------------------------
print("\n--- 3.2.2 Phân tích mức độ quan trọng của các đặc trưng ---")

# Chọn mô hình dựa trên cây tốt nhất hoặc RandomForest mặc định
model_for_importance = None
potential_tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'Decision Tree']

if best_model_name in potential_tree_models:
    model_for_importance = models[best_model_name]
    print(f"Sử dụng mô hình '{best_model_name}' để phân tích Feature Importance.")
elif 'Random Forest' in models:
    model_for_importance = models['Random Forest']
    print(f"Mô hình tốt nhất ({best_model_name}) không phải dạng cây. Sử dụng 'Random Forest' để phân tích Feature Importance.")
else:
    print("Không tìm thấy mô hình phù hợp (dạng cây) để trích xuất feature importance.")

if model_for_importance and hasattr(model_for_importance, 'feature_importances_'):
    importances = model_for_importance.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names_processed,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nMức độ quan trọng của các đặc trưng (Top 20):")
    print(feature_importance_df.head(20))

    # Trực quan hóa
    top_n = 20
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature',
                data=feature_importance_df.head(top_n),
                palette='magma') # Đổi màu palette
    plt.title(f'Top {top_n} Đặc trưng quan trọng nhất (Mô hình: {type(model_for_importance).__name__})')
    plt.xlabel('Mức độ quan trọng (Importance Score)')
    plt.ylabel('Đặc trưng (Feature)')
    plt.tight_layout()
    plt.show()
else:
     # Nếu không có feature importance (ví dụ: mô hình tuyến tính là tốt nhất)
     # Có thể xem xét hệ số của mô hình tuyến tính (nhưng cần cẩn thận vì đã scale)
     if best_model_name in ['Linear Regression', 'Ridge', 'Lasso']:
         try:
             coefficients = models[best_model_name].coef_
             coef_df = pd.DataFrame({
                 'Feature': all_feature_names_processed,
                 'Coefficient': coefficients
             }).sort_values(by='Coefficient', key=abs, ascending=False) # Sắp xếp theo giá trị tuyệt đối
             print("\nHệ số của mô hình tuyến tính (đã chuẩn hóa, độ lớn thể hiện tầm quan trọng tương đối):")
             print(coef_df.head(20))
             # Lưu ý: Diễn giải hệ số cần cẩn thận do có one-hot encoding và scaling
         except Exception as e:
             print(f"Không thể lấy hệ số từ mô hình {best_model_name}: {e}")
     else:
        print("Không thể thực hiện phân tích Feature Importance cho mô hình tốt nhất.")


# ------------------------------------------------------------------------------
# 3.2.3 (Tùy chọn) Loại bỏ đặc trưng ít quan trọng và huấn luyện lại mô hình
# ------------------------------------------------------------------------------
print("\n--- 3.2.3 (Tùy chọn) Loại bỏ đặc trưng ít quan trọng ---")

run_feature_selection = True # Đặt thành False nếu không muốn chạy
importance_threshold = 0.001 # Ngưỡng để giữ lại features (có thể điều chỉnh)
X_train_final = X_train_processed_df.copy() # Khởi tạo giá trị mặc định
X_test_final = X_test_processed_df.copy()   # Khởi tạo giá trị mặc định

if run_feature_selection and 'feature_importance_df' in locals() and not feature_importance_df.empty:
    important_features = feature_importance_df[feature_importance_df['Importance'] >= importance_threshold]['Feature'].tolist()
    removed_features = feature_importance_df[feature_importance_df['Importance'] < importance_threshold]['Feature'].tolist()

    print(f"Số lượng features ban đầu: {X_train_processed_df.shape[1]}")
    print(f"Ngưỡng importance để giữ lại: {importance_threshold}")
    print(f"Số lượng features giữ lại: {len(important_features)}")
    print(f"Số lượng features bị loại bỏ: {len(removed_features)}")
    # print(f"Các features bị loại bỏ: {removed_features}") # Bỏ comment nếu muốn xem chi tiết

    if len(important_features) < X_train_processed_df.shape[1] and len(important_features) > 0:
        # Tạo bộ dữ liệu mới với các feature quan trọng
        X_train_important = X_train_processed_df[important_features]
        X_test_important = X_test_processed_df[important_features]

        print(f"\nHuấn luyện lại mô hình '{best_model_name}' với {len(important_features)} features quan trọng...")
        # Lấy lại instance mới của mô hình tốt nhất để tránh fit chồng lên nhau
        model_refit_config = models[best_model_name].get_params()
        model_refit = type(models[best_model_name])(**model_refit_config)
        if 'n_jobs' in model_refit_config: model_refit.set_params(n_jobs=-1) # Đảm bảo n_jobs
        if 'verbosity' in model_refit_config: model_refit.set_params(verbosity=-1) # Đảm bảo verbosity


        start_time_refit = time.time()
        model_refit.fit(X_train_important, y_train)
        y_pred_refit = model_refit.predict(X_test_important)
        end_time_refit = time.time()

        mae_refit = mean_absolute_error(y_test, y_pred_refit)
        rmse_refit = np.sqrt(mean_squared_error(y_test, y_pred_refit))
        r2_refit = r2_score(y_test, y_pred_refit)

        print(f"Kết quả mô hình '{best_model_name}' sau khi loại bỏ features:")
        print(f"  MAE: {mae_refit:,.0f} VND")
        print(f"  RMSE: {rmse_refit:,.0f} VND")
        print(f"  R2 Score: {r2_refit:.4f}")
        print(f"  Thời gian huấn luyện: {end_time_refit - start_time_refit:.2f} giây")

        # So sánh và quyết định
        original_r2 = results[best_model_name]['R2 Score']
        if r2_refit >= original_r2 - 0.005: # Cho phép giảm nhẹ R2 (ví dụ 0.005)
             print("Hiệu suất không giảm đáng kể. Sử dụng bộ features đã lọc cho bước tiếp theo.")
             X_train_final = X_train_important
             X_test_final = X_test_important
        else:
             print("Hiệu suất giảm đáng kể. Giữ lại bộ features gốc cho bước tiếp theo.")
             # X_train_final, X_test_final đã được gán giá trị gốc ở đầu
    elif len(important_features) == 0:
        print("Không có features nào vượt ngưỡng quan trọng. Giữ lại bộ features gốc.")
    else:
        print("Không có feature nào bị loại bỏ với ngưỡng hiện tại. Giữ lại bộ features gốc.")
else:
    print("Bỏ qua bước loại bỏ feature hoặc không có thông tin importance.")
    # Đảm bảo X_train_final, X_test_final giữ nguyên giá trị ban đầu (đã xử lý)


# ------------------------------------------------------------------------------
# 3.2.4 Tối ưu hóa siêu tham số (Optuna) và đánh giá mô hình cuối cùng
# ------------------------------------------------------------------------------
print("\n--- 3.2.4 Tối ưu hóa siêu tham số và đánh giá mô hình cuối cùng ---")

model_to_optimize_name = best_model_name # Tối ưu mô hình tốt nhất đã tìm thấy
print(f"Tiến hành tối ưu hóa siêu tham số cho mô hình: {model_to_optimize_name}")

def objective(trial):
    """Hàm mục tiêu cho Optuna, trả về RMSE trung bình từ Cross-Validation."""
    if model_to_optimize_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 600, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 60, step=5), # Có thể dùng log=True nếu muốn
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
    elif model_to_optimize_name == 'Gradient Boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)
    elif model_to_optimize_name == 'XGBoost':
         params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 18),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1
        }
         model = XGBRegressor(**params)
    elif model_to_optimize_name == 'LightGBM':
        params = {
            'objective': 'regression_l1', # MAE objective, có thể dùng 'regression' (L2)
            'metric': 'rmse', # Metric để theo dõi
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        }
        model = LGBMRegressor(**params)
    # Thêm các mô hình khác nếu cần tối ưu
    elif model_to_optimize_name in ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree']:
         print(f"Tối ưu hóa tự động cho {model_to_optimize_name} ít phổ biến hoặc phức tạp hơn.")
         print("Sử dụng cấu hình gốc.")
         # Trả về RMSE của mô hình gốc để Optuna không chọn nó nếu có mô hình khác tốt hơn
         model = models[model_to_optimize_name]
         model.fit(X_train_final, y_train) # Fit lại để đảm bảo
         y_pred_base = model.predict(X_train_final) # Đánh giá trên train để Optuna so sánh
         return np.sqrt(mean_squared_error(y_train, y_pred_base))
    else:
        print(f"Tối ưu hóa cho {model_to_optimize_name} chưa được định nghĩa.")
        return float('inf') # Giá trị lớn để không được chọn

    # Đánh giá bằng Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Sử dụng neg_root_mean_squared_error vì Optuna tối thiểu hóa
    scores = cross_val_score(model, X_train_final, y_train,
                             scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
    rmse_cv = -np.mean(scores)

    return rmse_cv

# Chạy Optuna Study
print(f"Bắt đầu quá trình tối ưu hóa (có thể mất vài phút)...")
study = optuna.create_study(direction='minimize') # Mục tiêu là giảm thiểu RMSE
n_trials = 50 # Số lần thử siêu tham số (tăng lên nếu có thời gian, ví dụ 100, 200)
study.optimize(objective, n_trials=n_trials, timeout=600) # Thêm timeout (vd: 10 phút)

print(f"\nHoàn tất tối ưu hóa siêu tham số sau {len(study.trials)} lần thử.")
print("Siêu tham số tốt nhất tìm được:")
best_params = study.best_params
print(best_params)
best_rmse_cv = study.best_value
print(f"Giá trị RMSE tốt nhất trên Cross-Validation: {best_rmse_cv:,.0f} VND")

# --- Huấn luyện và đánh giá mô hình cuối cùng ---
print("\nHuấn luyện mô hình cuối cùng với siêu tham số tối ưu...")

# Khởi tạo lại mô hình tốt nhất với các tham số tối ưu
final_model = None
if model_to_optimize_name == 'Random Forest':
    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
elif model_to_optimize_name == 'Gradient Boosting':
    final_model = GradientBoostingRegressor(**best_params, random_state=42)
elif model_to_optimize_name == 'XGBoost':
     final_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42, n_jobs=-1)
elif model_to_optimize_name == 'LightGBM':
     final_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=-1)
else:
     # Nếu mô hình không được tối ưu, sử dụng mô hình tốt nhất ban đầu
     print(f"Sử dụng cấu hình gốc tốt nhất cho {model_to_optimize_name}")
     final_model = models[model_to_optimize_name]


start_time_final = time.time()
final_model.fit(X_train_final, y_train) # Huấn luyện trên toàn bộ tập train đã xử lý/lọc
end_time_final = time.time()
y_pred_final = final_model.predict(X_test_final) # Đánh giá trên tập test cuối cùng

# Đánh giá cuối cùng trên tập Test
mae_final = mean_absolute_error(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
r2_final = r2_score(y_test, y_pred_final)

print("\n--- Kết quả đánh giá mô hình cuối cùng trên tập Test ---")
print(f"  Mô hình: {type(final_model).__name__}")
print(f"  Siêu tham số: {final_model.get_params()}") # In ra các tham số cuối cùng
print("-" * 20)
print(f"  MAE: {mae_final:,.0f} VND")
print(f"  RMSE: {rmse_final:,.0f} VND")
print(f"  R2 Score: {r2_final:.4f}")
print(f"  Thời gian huấn luyện cuối cùng: {end_time_final - start_time_final:.2f} giây")

# So sánh với kết quả trước khi tối ưu
print("\nSo sánh với mô hình gốc tốt nhất (trước tối ưu):")
original_best_results = results[best_model_name]
print(f"  R2 gốc: {original_best_results['R2 Score']:.4f}  -> R2 sau tối ưu: {r2_final:.4f}")
print(f"  RMSE gốc: {original_best_results['RMSE (VND)']:,.0f} -> RMSE sau tối ưu: {rmse_final:,.0f}")
print(f"  MAE gốc: {original_best_results['MAE (VND)']:,.0f}  -> MAE sau tối ưu: {mae_final:,.0f}")


# --- Trực quan hóa kết quả cuối cùng ---
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred_final, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Dự đoán hoàn hảo')
plt.xlabel("Giá thực tế (VND)")
plt.ylabel("Giá dự đoán (VND)")
plt.title(f"Giá thực tế vs. Giá dự đoán (Mô hình: {type(final_model).__name__}, R2={r2_final:.4f})")
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Biểu đồ phân phối lỗi (Residuals)
residuals = y_test - y_pred_final
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Lỗi dự đoán (Residuals = Actual - Predicted)")
plt.ylabel("Tần suất")
plt.title("Phân phối lỗi dự đoán của mô hình cuối cùng")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()

print("\nQuy trình xây dựng mô hình Machine Learning hoàn chỉnh đã thực hiện xong.")