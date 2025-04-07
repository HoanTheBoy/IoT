# -*- coding: utf-8 -*-
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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import re # Có thể cần cho làm sạch text phức tạp hơn

# Cài đặt hiển thị cho Pandas và Matplotlib/Seaborn
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("--- CHƯƠNG 3: QUY TRÌNH XÂY DỰNG MÔ HÌNH MACHINE LEARNING ---")

# --- 3.1 Thu thập và tiền xử lý dữ liệu laptop ---
print("\n--- 3.1 Thu thập và tiền xử lý dữ liệu ---")

# 3.1.1 Thu thập dữ liệu
print("\n--- 3.1.1 Thu thập dữ liệu ---")
try:
    # Đảm bảo file 'dataset11.csv' nằm cùng thư mục hoặc cung cấp đường dẫn đúng
    df = pd.read_csv('dataset11.csv')
    print(f"Đã tải thành công dữ liệu từ 'dataset11.csv'.")
    print(f"Kích thước dữ liệu: {df.shape[0]} hàng, {df.shape[1]} cột.")
    print("5 dòng dữ liệu đầu tiên:")
    print(df.head())
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'dataset11.csv'. Vui lòng kiểm tra lại đường dẫn.")
    exit() # Thoát nếu không đọc được file

# --- 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA) ---
print("\n--- 3.1.2 Phân tích thống kê và trực quan hóa dữ liệu (EDA) ---")

# Thông tin cơ bản
print("\nThông tin tổng quan về dữ liệu:")
df.info()

# Kiểm tra giá trị thiếu
print("\nKiểm tra giá trị thiếu:")
print(df.isnull().sum())
# --> Không có giá trị thiếu trong bộ dữ liệu này.

# Mô tả thống kê cho các cột số
print("\nMô tả thống kê cho các cột số:")
print(df.describe())

# Phân tích biến mục tiêu (Price_VND)
plt.figure(figsize=(10, 5))
sns.histplot(df['Price_VND'], kde=True, bins=30) # Tăng bins để xem chi tiết hơn
plt.title('Phân phối giá Laptop (Price_VND)')
plt.xlabel('Giá (VND)')
plt.ylabel('Số lượng')
plt.ticklabel_format(style='plain', axis='x') # Tắt định dạng khoa học
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Nhận xét phân phối giá: Phân phối lệch phải.")

# Phân tích các biến Categorical (chỉ vẽ một vài ví dụ)
print("\nPhân tích ví dụ một số biến Categorical:")
categorical_cols_eda = ['Segment', 'CPU_Vendor', 'RAM_Type', 'GPU_Type', 'Screen_Panel', 'Design', 'Case_Material', 'Keyboard_Backlight']
for col in categorical_cols_eda:
    if col in df.columns:
        plt.figure(figsize=(10, 4))
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order)
        plt.title(f'Phân phối của {col}')
        plt.xlabel('Số lượng')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

# Phân tích các biến Numerical (chỉ vẽ một vài ví dụ)
print("\nPhân tích ví dụ một số biến Numerical:")
numerical_cols_eda = ['RAM_GB', 'Storage_GB', 'GPU_VRAM_GB', 'Screen_Size', 'Screen_Refresh', 'Battery_Whr']
for col in numerical_cols_eda:
     if col in df.columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=False)
        plt.title(f'Phân phối của {col}')
        plt.xlabel(col)
        plt.ylabel('Số lượng')
        plt.tight_layout()
        plt.show()

# Phân tích mối quan hệ giữa các biến và giá
print("\nPhân tích mối quan hệ giữa các biến và giá (ví dụ):")

# Biến Categorical vs Giá (ví dụ: Segment)
plt.figure(figsize=(10, 5))
order = df.groupby('Segment')['Price_VND'].median().sort_values().index
sns.boxplot(y='Segment', x='Price_VND', data=df, order=order)
plt.title(f'Giá theo Segment')
plt.xlabel('Giá (VND)')
plt.ylabel('Segment')
plt.ticklabel_format(style='plain', axis='x')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Biến Numerical vs Giá (ví dụ: RAM_GB)
plt.figure(figsize=(10, 5))
sns.scatterplot(x='RAM_GB', y='Price_VND', data=df, alpha=0.5)
plt.title(f'Giá vs RAM_GB')
plt.xlabel('RAM_GB')
plt.ylabel('Giá (VND)')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.show()

# Ma trận tương quan giữa các biến số
print("\nMa trận tương quan giữa các biến số và giá:")
plt.figure(figsize=(12, 10))
numerical_with_target = df.select_dtypes(include=np.number).columns.tolist()
correlation_matrix = df[numerical_with_target].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Ma trận tương quan giữa các biến số và giá')
plt.tight_layout()
plt.show()


# --- 3.1.3 Tiền xử lý dữ liệu ---
print("\n--- 3.1.3 Tiền xử lý dữ liệu ---")

# Sao chép DataFrame để tránh thay đổi bản gốc
df_processed = df.copy()

# (a) Làm sạch dữ liệu text (CPU, GPU, RAM,...)
# Dữ liệu có vẻ đã khá sạch, nhưng ví dụ nếu cần chuẩn hóa:
# df_processed['Case_Material'] = df_processed['Case_Material'].str.lower().str.replace('_', ' ')
print("Kiểm tra các cột text (ví dụ: CPU_Model, GPU_Name): Dữ liệu có vẻ đã được chuẩn hóa.")
# Xử lý cụ thể cho CPU_Generation: coi là categorical
df_processed['CPU_Generation'] = df_processed['CPU_Generation'].astype(str)
print("Đã chuyển đổi 'CPU_Generation' sang kiểu string để xử lý như categorical.")

# (b) Feature Engineering (Ví dụ)
# Có thể tạo các biến mới, ví dụ:
# df_processed['Density_Storage'] = df_processed['Storage_GB'] / df_processed['Screen_Size']
# Tạm thời không thêm feature mới phức tạp trong ví dụ này.
print("Bước Feature Engineering: Tạm thời không tạo đặc trưng mới.")


# Xác định các cột features (X) và target (y)
X = df_processed.drop('Price_VND', axis=1)
y = df_processed['Price_VND']

# Xác định lại các cột numerical và categorical sau khi làm sạch/engineering
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nCác cột numerical features cuối cùng: {numerical_features}")
print(f"\nCác cột categorical features cuối cùng: {categorical_features}")


# (c) Mã hóa biến Categorical & Chuẩn hóa biến Numerical bằng Pipeline
# Sử dụng OneHotEncoder cho biến categorical và StandardScaler cho biến numerical
# handle_unknown='ignore' để xử lý các giá trị mới trong test set
# sparse_output=False để dễ làm việc với kết quả OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' # Bỏ qua các cột không được chỉ định (nên là mặc định nếu đã xác định hết)
)
print("\nĐã định nghĩa pipeline tiền xử lý (StandardScaler cho số, OneHotEncoder cho categorical).")

# --- 3.2 Xây dựng mô hình Machine Learning (Hồi quy) ---
print("\n--- 3.2 Xây dựng mô hình Machine Learning (Hồi quy) ---")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nĐã chia dữ liệu: {X_train.shape[0]} mẫu huấn luyện, {X_test.shape[0]} mẫu kiểm tra.")

# --- 3.2.1 Huấn luyện và đánh giá các mô hình Machine Learning (Hồi quy) ---
print("\n--- 3.2.1 Huấn luyện và đánh giá các mô hình cơ bản ---")

# Định nghĩa các mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=2000), # Tăng max_iter cho Lasso
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Support Vector Regressor": SVR(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "XGBoost Regressor": xgb.XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM Regressor": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1) # Tắt bớt thông báo
}

results = {}

# Huấn luyện và đánh giá từng mô hình
for name, model in models.items():
    print(f"\nĐang huấn luyện mô hình: {name}")
    # Tạo pipeline hoàn chỉnh: Tiền xử lý -> Mô hình
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Huấn luyện
    try:
        pipeline.fit(X_train, y_train)

        # Dự đoán
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Đánh giá
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        results[name] = {
            "RMSE_Train": rmse_train,
            "MAE_Train": mae_train,
            "R2_Train": r2_train,
            "RMSE_Test": rmse_test,
            "MAE_Test": mae_test,
            "R2_Test": r2_test
        }

        print(f"Kết quả cho {name}:")
        print(f"  RMSE (Train): {rmse_train:,.2f} VND | R2 (Train): {r2_train:.4f}")
        # print(f"  MAE (Train):  {mae_train:,.2f} VND")
        print(f"  RMSE (Test):  {rmse_test:,.2f} VND | R2 (Test):  {r2_test:.4f}")
        # print(f"  MAE (Test):   {mae_test:,.2f} VND")

    except Exception as e:
        print(f"Lỗi khi huấn luyện hoặc đánh giá mô hình {name}: {e}")
        results[name] = {"RMSE_Train": np.nan, "MAE_Train": np.nan, "R2_Train": np.nan,
                         "RMSE_Test": np.nan, "MAE_Test": np.nan, "R2_Test": np.nan}


# So sánh kết quả các mô hình
results_df = pd.DataFrame(results).T.sort_values(by="R2_Test", ascending=False)
print("\nBảng tổng hợp kết quả (sắp xếp theo R2 Test giảm dần):")
print(results_df[['R2_Train', 'R2_Test', 'RMSE_Test', 'MAE_Test']].to_string(float_format='{:,.2f}'.format)) # Format cho dễ đọc

best_model_name = results_df.index[0]
print(f"\nMô hình có kết quả tốt nhất trên tập test (dựa trên R2): {best_model_name}")

# --- 3.2.2 Phân tích mức độ quan trọng của các đặc trưng (Thông số kỹ thuật) ---
print("\n--- 3.2.2 Phân tích mức độ quan trọng của các đặc trưng ---")

# Lấy mô hình tốt nhất từ kết quả trên (hoặc chọn một mô hình cây như RF, GB, XGB, LGBM)
# Ưu tiên mô hình ensemble để có feature importance
model_for_importance = None
if best_model_name in ["Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor"]:
    model_for_importance = models[best_model_name]
    print(f"Sử dụng mô hình '{best_model_name}' để phân tích Feature Importance.")
else:
    # Nếu mô hình tốt nhất không phải dạng cây, dùng RandomForest làm đại diện
    model_for_importance = RandomForestRegressor(random_state=42, n_jobs=-1)
    print(f"Mô hình tốt nhất ({best_model_name}) không có feature importance mặc định. Sử dụng RandomForestRegressor thay thế để phân tích.")

# Tạo pipeline chỉ với mô hình được chọn để lấy importance
pipeline_importance = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', model_for_importance)])

print("Đang huấn luyện lại mô hình được chọn để lấy Feature Importance...")
try:
    pipeline_importance.fit(X_train, y_train)
    print("Huấn luyện hoàn tất.")

    # Lấy tên đặc trưng sau khi biến đổi (OneHotEncoder)
    feature_names_out = pipeline_importance.named_steps['preprocessor'].get_feature_names_out()
    importances = pipeline_importance.named_steps['regressor'].feature_importances_

    # Tạo DataFrame để xem và vẽ biểu đồ
    feature_importance_df = pd.DataFrame({'Feature': feature_names_out, 'Importance': importances})
    # Chuẩn hóa Importance để tổng bằng 100 (dễ so sánh hơn)
    feature_importance_df['Importance (%)'] = (feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()) * 100
    feature_importance_df = feature_importance_df.sort_values(by='Importance (%)', ascending=False).reset_index(drop=True)

    print(f"\nBảng xếp hạng mức độ quan trọng của các đặc trưng (Mô hình: {type(model_for_importance).__name__}):")
    # In ra toàn bộ hoặc top N
    print(feature_importance_df[['Feature', 'Importance (%)']].round(4).to_string()) # In toàn bộ, làm tròn

    # Vẽ biểu đồ top N đặc trưng quan trọng
    n_top_features = 30 # Số lượng đặc trưng hàng đầu muốn hiển thị
    plt.figure(figsize=(12, max(8, n_top_features * 0.3))) # Điều chỉnh kích thước biểu đồ
    sns.barplot(x='Importance (%)', y='Feature', data=feature_importance_df.head(n_top_features), palette='viridis')
    plt.title(f'Top {n_top_features} đặc trưng quan trọng nhất ({type(model_for_importance).__name__})')
    plt.xlabel('Mức độ quan trọng (%)')
    plt.ylabel('Đặc trưng (Sau tiền xử lý)')
    plt.tight_layout()
    plt.show()

except Exception as e:
        print(f"Lỗi khi lấy hoặc xử lý Feature Importance: {e}")

print("\n--- KẾT THÚC CHƯƠNG 3 ---")