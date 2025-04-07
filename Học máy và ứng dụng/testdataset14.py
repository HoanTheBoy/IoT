# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) # Default figure size

# --- CHƯƠNG 3: QUY TRÌNH XÂY DỰNG MÔ HÌNH MACHINE LEARNING ---

print("--- CHƯƠNG 3: QUY TRÌNH XÂY DỰNG MÔ HÌNH MACHINE LEARNING ---")

# --- 3.1 Thu thập và tiền xử lý dữ liệu laptop ---
print("\n--- 3.1 Thu thập và tiền xử lý dữ liệu laptop ---")

# 3.1.1 Thu thập dữ liệu (từ file dataset14.csv)
print("\n### 3.1.1 Thu thập dữ liệu ###")
try:
    df = pd.read_csv('dataset14.csv')
    print(f"Đã tải thành công dữ liệu từ 'dataset14.csv'.")
    print(f"Bộ dữ liệu có {df.shape[0]} hàng và {df.shape[1]} cột.")
    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head())
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'dataset14.csv'. Vui lòng đảm bảo file tồn tại trong cùng thư mục.")
    exit() # Exit if data cannot be loaded

# 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA)
print("\n### 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA) ###")

# --- Thông tin cơ bản ---
print("\n--- Thông tin cơ bản về dữ liệu (dtypes, non-null counts) ---")
df.info()

# --- Thống kê mô tả cho các biến số ---
print("\n--- Thống kê mô tả (biến số) ---")
# Convert price to millions for easier description interpretation
df['Price_Millions'] = df['Price_VND'] / 1_000_000
print(df[['RAM_GB', 'Storage_GB', 'GPU_VRAM_GB', 'Screen_Size', 'Screen_Refresh', 'Battery_Whr', 'Price_Millions']].describe())
df = df.drop('Price_Millions', axis=1) # Drop temporary column

# --- Kiểm tra giá trị thiếu ---
print("\n--- Kiểm tra giá trị thiếu ---")
print(df.isnull().sum())
# No missing values found in this dataset based on the output of df.info()

# --- Phân tích biến mục tiêu (Price_VND) ---
print("\n--- Phân tích biến mục tiêu (Price_VND) ---")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Price_VND'], kde=True, bins=30)
plt.title('Phân phối Giá Laptop (VND)')
plt.xlabel('Giá (VND)')
plt.ylabel('Tần suất')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Price_VND'])
plt.title('Boxplot Giá Laptop (VND)')
plt.xlabel('Giá (VND)')

plt.tight_layout()
plt.show()

# Log transform Price for potentially better model performance due to skewness
plt.figure(figsize=(12, 5))
df['Price_VND_Log'] = np.log1p(df['Price_VND']) # log1p handles potential zero values

plt.subplot(1, 2, 1)
sns.histplot(df['Price_VND_Log'], kde=True, bins=30)
plt.title('Phân phối Log(Giá Laptop + 1)')
plt.xlabel('Log(Giá + 1)')
plt.ylabel('Tần suất')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Price_VND_Log'])
plt.title('Boxplot Log(Giá Laptop + 1)')
plt.xlabel('Log(Giá + 1)')

plt.tight_layout()
plt.show()
print("Nhận xét: Phân phối giá gốc bị lệch phải (right-skewed). Phân phối log(Giá) gần với phân phối chuẩn hơn.")

# --- Phân tích các biến số (Numerical Features) ---
print("\n--- Phân tích các biến số (Numerical Features) ---")
numerical_features = ['RAM_GB', 'Storage_GB', 'GPU_VRAM_GB', 'Screen_Size', 'Screen_Refresh', 'Battery_Whr']
df[numerical_features].hist(bins=20, figsize=(15, 10))
plt.suptitle('Phân phối các Biến Số')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()

# --- Phân tích các biến phân loại (Categorical Features) ---
print("\n--- Phân tích các biến phân loại (Categorical Features) ---")
categorical_features = ['Product_Line', 'Segment', 'CPU_Vendor', 'CPU_Model', 'CPU_Generation',
                        'RAM_Type', 'GPU_Type', 'GPU_Name', 'Screen_Panel', 'Design',
                        'Case_Material', 'Keyboard_Backlight']

# Check cardinality (number of unique values)
print("\nSố lượng giá trị duy nhất trong các biến phân loại:")
print(df[categorical_features].nunique())

# Visualize some key categorical features
plt.figure(figsize=(18, 15))
plt.subplot(3, 3, 1)
sns.countplot(y=df['Product_Line'], order = df['Product_Line'].value_counts().index)
plt.title('Số lượng theo Dòng Sản phẩm')

plt.subplot(3, 3, 2)
sns.countplot(y=df['Segment'], order = df['Segment'].value_counts().index)
plt.title('Số lượng theo Phân khúc')

plt.subplot(3, 3, 3)
sns.countplot(y=df['CPU_Vendor'], order = df['CPU_Vendor'].value_counts().index)
plt.title('Số lượng theo Nhà cung cấp CPU')

plt.subplot(3, 3, 4)
sns.countplot(y=df['RAM_Type'], order = df['RAM_Type'].value_counts().index)
plt.title('Số lượng theo Loại RAM')

plt.subplot(3, 3, 5)
sns.countplot(y=df['GPU_Type'], order = df['GPU_Type'].value_counts().index)
plt.title('Số lượng theo Loại GPU')

plt.subplot(3, 3, 6)
sns.countplot(y=df['Screen_Panel'], order = df['Screen_Panel'].value_counts().index)
plt.title('Số lượng theo Loại Panel Màn hình')

plt.subplot(3, 3, 7)
sns.countplot(y=df['Case_Material'], order = df['Case_Material'].value_counts().index)
plt.title('Số lượng theo Vật liệu Vỏ')

plt.subplot(3, 3, 8)
sns.countplot(y=df['Keyboard_Backlight'], order = df['Keyboard_Backlight'].value_counts().index)
plt.title('Số lượng theo Đèn nền Bàn phím')

plt.subplot(3, 3, 9)
sns.countplot(y=df['Design'], order = df['Design'].value_counts().index)
plt.title('Số lượng theo Thiết kế')

plt.tight_layout()
plt.show()

# --- Phân tích mối quan hệ giữa các biến và giá ---
print("\n--- Phân tích mối quan hệ giữa các biến và Giá ---")

# Numerical features vs Price
plt.figure(figsize=(18, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=df[col], y=df['Price_VND'])
    plt.title(f'{col} vs Price_VND')
    plt.ylabel('Giá (VND)')
plt.tight_layout()
plt.show()

# Categorical features vs Price (using Log Price for better visualization)
plt.figure(figsize=(15, 25))
key_categoricals_for_price = ['Product_Line', 'Segment', 'CPU_Vendor', 'RAM_Type', 'GPU_Type', 'Screen_Panel', 'Case_Material', 'Keyboard_Backlight']
for i, col in enumerate(key_categoricals_for_price):
    plt.subplot(4, 2, i + 1)
    # Sort categories by median price for better visualization
    order = df.groupby(col)['Price_VND_Log'].median().sort_values().index
    sns.boxplot(y=df[col], x=df['Price_VND_Log'], order=order)
    plt.title(f'{col} vs Log(Price)')
    plt.xlabel('Log(Giá + 1)')
plt.tight_layout()
plt.show()

# --- Correlation Matrix ---
print("\n--- Ma trận tương quan (biến số) ---")
# Select only numerical columns for correlation calculation
numerical_cols_corr = df[numerical_features + ['Price_VND']].copy()
# Convert CPU_Generation to numeric if it's not already (it seems numeric but good practice)
# If CPU_Generation contains non-numeric strings (like '5000 series'), it needs cleaning first.
# Assuming CPU_Generation is already clean numeric for now.
numerical_cols_corr['CPU_Generation'] = pd.to_numeric(df['CPU_Generation'], errors='coerce')

plt.figure(figsize=(10, 8))
correlation_matrix = numerical_cols_corr.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận Tương quan giữa các Biến Số và Giá')
plt.show()

# 3.1.3 Tiền xử lý dữ liệu
print("\n### 3.1.3 Tiền xử lý dữ liệu ###")

# --- Làm sạch dữ liệu text / Feature Engineering ---
# Dựa trên EDA, các cột text chính không cần làm sạch nhiều.
# Tuy nhiên, CPU_Model và GPU_Name có quá nhiều giá trị duy nhất (high cardinality).
# -> Quyết định: Loại bỏ CPU_Model và GPU_Name để tránh tạo ra quá nhiều cột khi One-Hot Encoding.
#    Các thông tin quan trọng hơn có thể được nắm bắt bởi CPU_Vendor, CPU_Generation, GPU_Type, GPU_VRAM_GB.
# -> Biến CPU_Generation đã là số, có thể coi là numerical hoặc categorical (ordinal). Coi là numerical ở đây.
# -> Keyboard_Backlight (Yes/No) có thể map thành 1/0, nhưng OneHotEncoder cũng xử lý được.

# --- Xác định các loại biến ---
target = 'Price_VND_Log' # Sử dụng log-transformed price làm target
y = df[target]
# Drop original price, log price, and high cardinality features
X = df.drop(['Price_VND', 'Price_VND_Log', 'CPU_Model', 'GPU_Name'], axis=1)

numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print("\nBiến Số (Numerical):", numerical_features)
print("Biến Phân loại (Categorical):", categorical_features)

# --- Mã hóa biến Categorical & Chuẩn hóa biến Numerical ---
# Sử dụng Pipeline và ColumnTransformer

# Tạo preprocessor
# OneHotEncoder cho categorical features: handle_unknown='ignore' để xử lý các giá trị có thể xuất hiện trong test set nhưng không có trong train set.
# StandardScaler cho numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse=False easier for feature names
    ],
    remainder='passthrough' # Giữ lại các cột không được chỉ định (nếu có)
)

print("\nĐã tạo Preprocessor với StandardScaler cho biến số và OneHotEncoder cho biến phân loại.")

# --- Chia dữ liệu thành tập huấn luyện và tập kiểm tra ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nChia dữ liệu: {X_train.shape[0]} mẫu huấn luyện, {X_test.shape[0]} mẫu kiểm tra.")


# --- 3.2 Xây dựng mô hình Machine Learning (Hồi quy) ---
print("\n--- 3.2 Xây dựng mô hình Machine Learning (Hồi quy) ---")

# 3.2.1 Huấn luyện và đánh giá các mô hình Machine Learning (Hồi quy)
print("\n### 3.2.1 Huấn luyện và đánh giá các mô hình ###")

# Định nghĩa các mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Lasso Regression": Lasso(alpha=0.01, random_state=42), # Lower alpha for Lasso initially
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10), # Limit depth to prevent overfitting
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=5), # Add some regularization
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=5) # Common starting parameters
}

results = {}

# Huấn luyện và đánh giá từng mô hình
for name, model in models.items():
    print(f"\n--- Training {name} ---")

    # Tạo pipeline hoàn chỉnh
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Huấn luyện
    pipeline.fit(X_train, y_train)

    # Dự đoán trên tập test
    y_pred_log = pipeline.predict(X_test)

    # Chuyển đổi dự đoán về thang đo gốc
    y_pred_original = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test) # Also transform y_test back for original scale metrics

    # Đánh giá
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test, y_pred_log) # R2 thường được tính trên thang đo log nếu target là log

    results[name] = {'MAE (VND)': mae, 'RMSE (VND)': rmse, 'R2 (Log Scale)': r2}

    print(f"{name} - Đánh giá:")
    print(f"  MAE (VND): {mae:,.0f}")
    print(f"  RMSE (VND): {rmse:,.0f}")
    print(f"  R2 Score (Log Scale): {r2:.4f}")

# --- So sánh kết quả ---
print("\n--- So sánh kết quả các mô hình ---")
results_df = pd.DataFrame(results).T.sort_values(by='R2 (Log Scale)', ascending=False)
print(results_df)

best_model_name = results_df.index[0]
print(f"\nMô hình tốt nhất dựa trên R2 Score (Log Scale) là: {best_model_name}")

# 3.2.2 Phân tích mức độ quan trọng của các đặc trưng (Thông số kỹ thuật)
print("\n### 3.2.2 Phân tích mức độ quan trọng của các đặc trưng ###")

# Chọn mô hình tốt nhất để phân tích (Thường là Random Forest hoặc Gradient Boosting)
best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', models[best_model_name])]) # Rebuild the pipeline
best_model_pipeline.fit(X_train, y_train)

# Lấy tên các đặc trưng sau khi biến đổi
try:
    # Correct way to get feature names from ColumnTransformer within a Pipeline
    feature_names_out = best_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
except AttributeError:
     # Fallback for older scikit-learn versions or different preprocessor structures
     print("Could not automatically get feature names. Manual definition might be needed.")
     # Attempt manual reconstruction (might need adjustment based on exact preprocessor steps)
     ohe_feature_names = best_model_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
     feature_names_out = numerical_features + list(ohe_feature_names)


# Lấy độ quan trọng của đặc trưng từ mô hình
if hasattr(best_model_pipeline.named_steps['regressor'], 'feature_importances_'):
    importances = best_model_pipeline.named_steps['regressor'].feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print(f"\n--- Độ quan trọng đặc trưng từ mô hình {best_model_name} ---")
    print(feature_importance_df.head(15)) # Hiển thị top 15

    # Trực quan hóa độ quan trọng
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title(f'Top 15 Đặc trưng Quan trọng nhất ({best_model_name})')
    plt.xlabel('Độ quan trọng')
    plt.ylabel('Đặc trưng')
    plt.tight_layout()
    plt.show()

elif hasattr(best_model_pipeline.named_steps['regressor'], 'coef_'):
    # For linear models (Linear Regression, Ridge, Lasso)
    importances = best_model_pipeline.named_steps['regressor'].coef_
    feature_importance_df = pd.DataFrame({'Feature': feature_names_out, 'Coefficient': importances})
    # Use absolute value for ranking importance magnitude
    feature_importance_df['Abs_Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)


    print(f"\n--- Hệ số hồi quy từ mô hình {best_model_name} (Đã chuẩn hóa) ---")
    print(feature_importance_df[['Feature', 'Coefficient']].head(15)) # Display top 15 by absolute coefficient

    # Visualize coefficients
    plt.figure(figsize=(10, 8))
    # Plot actual coefficients to see direction of influence
    plot_data = feature_importance_df.head(15).sort_values(by='Coefficient', ascending=False)
    sns.barplot(x='Coefficient', y='Feature', data=plot_data, palette='vlag')
    plt.title(f'Top 15 Hệ số Hồi quy Quan trọng nhất ({best_model_name})')
    plt.xlabel('Hệ số (thang đo Log Giá)')
    plt.ylabel('Đặc trưng')
    plt.tight_layout()
    plt.show()
else:
    print(f"Không thể trích xuất độ quan trọng/hệ số từ mô hình {best_model_name}.")


print("\n--- Kết thúc quy trình ---")