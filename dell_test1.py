import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re # Thư viện để làm việc với regular expressions

# --- Tải Dữ Liệu ---
# Đảm bảo file datafake.csv nằm trong cùng thư mục với notebook
# hoặc cung cấp đường dẫn đầy đủ tới file.
try:
    df = pd.read_csv('datafake.csv')
    print("Tải dữ liệu thành công.")
    print("Thông tin ban đầu của dữ liệu:")
    df.info()
    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head())
    print("\nThống kê mô tả cho các cột số:")
    print(df.describe())
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file datafake.csv. Hãy đảm bảo file nằm đúng vị trí.")
    exit() # Thoát nếu không tải được file

# --- II. Phương Pháp Phân Tích và Xây dựng Mô hình ---

# --- Xử lý Dữ liệu (Preprocessing) ---

print("\n--- Bắt đầu Tiền xử lý dữ liệu ---")

# 1. Làm sạch dữ liệu và Chuyển đổi kiểu dữ liệu

# Price: Chuyển sang kiểu số, loại bỏ các dòng có giá trị lỗi hoặc thiếu
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True) # Loại bỏ các dòng không có giá trị Price hợp lệ
df['Price'] = df['Price'].astype(float)
print(f"Số lượng bản ghi sau khi xử lý cột Price: {len(df)}")

# Storage: Đảm bảo là kiểu số
df['Storage'] = pd.to_numeric(df['Storage'], errors='coerce')

# ScreenSize: Đảm bảo là kiểu số
df['ScreenSize'] = pd.to_numeric(df['ScreenSize'], errors='coerce')

# Weight: Đảm bảo là kiểu số
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')

# RAM: Trích xuất số GB từ chuỗi (ví dụ: '8GB' -> 8)
# Sử dụng regex để tìm số đứng trước 'GB' hoặc số đứng một mình (dự phòng)
df['RAM_GB'] = df['RAM'].astype(str).str.extract(r'(\d+)', expand=False)
df['RAM_GB'] = pd.to_numeric(df['RAM_GB'], errors='coerce')
df.drop('RAM', axis=1, inplace=True) # Bỏ cột RAM gốc

# CPU: Trích xuất thông tin cơ bản (ví dụ: i3, i5, i7, Ryzen 5, Ultra 7)
# Đây là một bước đơn giản hóa, có thể cải thiện thêm sau này
def extract_cpu_type(cpu_str):
    cpu_str = str(cpu_str).lower()
    if 'ultra 7' in cpu_str: return 'Ultra 7'
    if 'ultra 5' in cpu_str: return 'Ultra 5'
    if 'i9' in cpu_str: return 'Core i9'
    if 'i7' in cpu_str: return 'Core i7'
    if 'i5' in cpu_str: return 'Core i5'
    if 'i3' in cpu_str: return 'Core i3'
    if 'ryzen 7' in cpu_str: return 'Ryzen 7'
    if 'ryzen 5' in cpu_str: return 'Ryzen 5'
    if 'ryzen 3' in cpu_str: return 'Ryzen 3'
    if 'core 7' in cpu_str: return 'Core 7' # Dòng CPU mới của Intel
    if 'core 5' in cpu_str: return 'Core 5' # Dòng CPU mới của Intel
    return 'Other' # Các loại khác hoặc không xác định rõ

df['CPU_Type'] = df['CPU'].apply(extract_cpu_type)
df.drop('CPU', axis=1, inplace=True)

# GPU: Phân loại cơ bản (Intel, NVIDIA, AMD) và có card rời hay không
def classify_gpu(gpu_str):
    gpu_str = str(gpu_str).lower()
    is_dedicated = 0
    brand = 'Intel' # Mặc định là Intel (thường là UHD/Iris Xe tích hợp)
    if 'nvidia' in gpu_str or 'rtx' in gpu_str or 'geforce' in gpu_str or 'mx' in gpu_str:
        brand = 'NVIDIA'
        is_dedicated = 1
    elif 'amd' in gpu_str or 'radeon' in gpu_str:
        brand = 'AMD'
        # AMD Radeon Graphics có thể là tích hợp hoặc rời (ít phổ biến hơn trên Dell gần đây)
        # Coi là tích hợp trừ khi có tên cụ thể như RX series (không có trong data này)
        if 'radeon graphics' not in gpu_str:
             is_dedicated = 1 # Giả định nếu không phải 'Radeon Graphics' thì là card rời
    # Trích xuất VRAM nếu có (đơn giản hóa)
    vram_match = re.search(r'(\d+)\s?gb', gpu_str)
    vram = int(vram_match.group(1)) if vram_match else 0
    # Nếu VRAM > 0 và là card Intel -> có thể là Intel Arc, coi là dedicated
    if brand == 'Intel' and vram > 0:
        is_dedicated = 1

    return brand, is_dedicated, vram

gpu_info = df['GPU'].apply(classify_gpu)
df['GPU_Brand'] = gpu_info.apply(lambda x: x[0])
df['GPU_Is_Dedicated'] = gpu_info.apply(lambda x: x[1])
df['GPU_VRAM_GB'] = gpu_info.apply(lambda x: x[2])

df.drop('GPU', axis=1, inplace=True)

# Shell: Chuẩn hóa và xử lý (ví dụ: 'Plastic-Metal' -> 'Hybrid')
def standardize_shell(shell_str):
    shell_str = str(shell_str).lower()
    if 'metal' in shell_str and 'plastic' in shell_str: return 'Hybrid'
    if 'metal' in shell_str or 'aluminum' in shell_str: return 'Metal'
    if 'plastic' in shell_str: return 'Plastic'
    if 'carbon' in shell_str: return 'Carbon' # Thêm Carbon
    return 'Other'

df['Shell_Type'] = df['Shell'].apply(standardize_shell)
df.drop('Shell', axis=1, inplace=True)

# Battery: Trích xuất số Cells và dung lượng Wh (nếu có)
def extract_battery_info(battery_str):
    battery_str = str(battery_str).lower()
    cells_match = re.search(r'(\d+)-cell', battery_str)
    wh_match = re.search(r'(\d+)\s?wh', battery_str)
    cells = int(cells_match.group(1)) if cells_match else np.nan
    wh = int(wh_match.group(1)) if wh_match else np.nan
    return cells, wh

battery_info = df['Battery'].apply(extract_battery_info)
df['Battery_Cells'] = battery_info.apply(lambda x: x[0])
df['Battery_Wh'] = battery_info.apply(lambda x: x[1])
df.drop('Battery', axis=1, inplace=True)

# Bỏ cột 'Name' vì quá đa dạng và khó để trích xuất thông tin hữu ích một cách nhất quán
# Thông tin quan trọng từ 'Name' (như model series) có thể đã phản ánh qua các cột khác (CPU, GPU,...)
df.drop('Name', axis=1, inplace=True)

# 2. Xử lý giá trị thiếu (Missing Values)
print("\nKiểm tra giá trị thiếu trước khi xử lý:")
print(df.isnull().sum())

# Điền giá trị thiếu cho các cột số bằng median (ít bị ảnh hưởng bởi outliers)
numerical_cols_with_na = df.select_dtypes(include=np.number).isnull().sum()
numerical_cols_with_na = numerical_cols_with_na[numerical_cols_with_na > 0].index.tolist()
for col in numerical_cols_with_na:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"Điền giá trị thiếu cột '{col}' bằng median: {median_val}")

# Kiểm tra lại giá trị thiếu
print("\nKiểm tra giá trị thiếu SAU khi xử lý:")
print(df.isnull().sum()) # Các cột object không có giá trị thiếu trong TH này

# 3. Mã hóa biến phân loại (Categorical Encoding)
# Sử dụng One-Hot Encoding cho các biến phân loại còn lại
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nCác cột phân loại cần mã hóa: {categorical_cols}")

# Kiểm tra số lượng giá trị duy nhất để tránh tạo quá nhiều cột mới
for col in categorical_cols:
    print(f"Số lượng giá trị duy nhất trong cột '{col}': {df[col].nunique()}")

df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=False) # drop_first để tránh đa cộng tuyến
print("\nKích thước dữ liệu sau khi mã hóa one-hot:")
print(df_processed.shape)
print("\nCác cột sau khi mã hóa:")
print(df_processed.columns.tolist())


# --- Khám phá Dữ liệu (Exploratory Data Analysis - EDA) ---
print("\n--- Bắt đầu Khám phá Dữ liệu (EDA) ---")

# 1. Phân tích biến mục tiêu (Price)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_processed['Price'], kde=True)
plt.title('Phân phối của Giá Laptop (Price)')
plt.xlabel('Giá (VND)')
plt.ylabel('Tần suất')

# Xem xét phân phối log của giá nếu bị lệch nhiều
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df_processed['Price']), kde=True) # log1p(x) = log(1+x) để xử lý giá trị 0 nếu có
plt.title('Phân phối của Log(1 + Price)')
plt.xlabel('Log(1 + Giá)')
plt.ylabel('Tần suất')

plt.tight_layout()
plt.show()
# Nhận xét: Phân phối giá có vẻ lệch phải (right-skewed). Sử dụng log giá có thể giúp mô hình hoạt động tốt hơn.
# Tuy nhiên, để đơn giản, tạm thời vẫn sử dụng giá gốc.

# 2. Phân tích tương quan giữa các biến số
plt.figure(figsize=(18, 12))
# Chọn các cột số gốc (trước one-hot) và các cột số mới tạo để xem tương quan cốt lõi
cols_for_corr = ['Price', 'Storage', 'ScreenSize', 'Weight', 'RAM_GB',
                 'GPU_Is_Dedicated', 'GPU_VRAM_GB', 'Battery_Cells', 'Battery_Wh']
# Lấy thêm các cột one-hot nếu muốn, nhưng heatmap sẽ rất lớn
# numerical_after_dummies = df_processed.select_dtypes(include=np.number).columns.tolist()
correlation_matrix = df_processed[cols_for_corr].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Biểu đồ nhiệt tương quan giữa các biến số chính')
plt.show()
# Nhận xét:
# - RAM_GB, GPU_Is_Dedicated, GPU_VRAM_GB, Battery_Wh có tương quan dương khá rõ với Price.
# - Storage cũng có tương quan dương.
# - Weight, ScreenSize có tương quan dương yếu hơn.
# - Battery_Cells có tương quan không rõ ràng hoặc yếu.

# 3. Biểu đồ phân tán giữa các biến số quan trọng và Price
numerical_features = ['Storage', 'ScreenSize', 'Weight', 'RAM_GB', 'GPU_VRAM_GB', 'Battery_Wh']
print("\nBiểu đồ phân tán giữa các biến số và Giá:")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(data=df_processed, x=col, y='Price', alpha=0.5)
    plt.title(f'{col} vs Price')
plt.tight_layout()
plt.show()
# Nhận xét: Có vẻ có mối quan hệ tuyến tính (hoặc ít nhất là đơn điệu tăng) giữa RAM_GB, GPU_VRAM_GB, Storage và Price.

# 4. Phân tích các biến phân loại gốc với Price (sử dụng boxplot)
categorical_features_original = ['CPU_Type', 'GPU_Brand', 'Shell_Type', 'Shop']
print("\nBiểu đồ hộp giữa các biến phân loại và Giá:")
plt.figure(figsize=(18, 10))
for i, col in enumerate(categorical_features_original):
    plt.subplot(2, 2, i + 1)
    # Sắp xếp các hộp theo giá trung vị để dễ nhìn hơn
    order = df.groupby(col)['Price'].median().sort_values().index
    sns.boxplot(data=df, x=col, y='Price', order=order)
    plt.title(f'{col} vs Price')
    plt.xticks(rotation=45, ha='right') # Xoay nhãn nếu cần
plt.tight_layout()
plt.show()
# Nhận xét:
# - CPU_Type: Giá tăng dần từ i3 -> i5 -> i7 -> i9/Ultra.
# - GPU_Brand: NVIDIA thường có giá cao hơn Intel/AMD (trung vị).
# - Shell_Type: Metal và Carbon thường có giá cao hơn Plastic.
# - Shop: Có sự khác biệt về phân phối giá giữa các cửa hàng.

# --- Xây dựng Mô hình Hồi quy Tuyến tính ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm # Để xem chi tiết thống kê mô hình

print("\n--- Bắt đầu Xây dựng Mô hình Hồi quy Tuyến tính ---")

# 1. Lựa chọn biến
# Biến phụ thuộc (Target)
y = df_processed['Price']
# Biến độc lập (Features) - loại bỏ biến mục tiêu ban đầu
X = df_processed.drop('Price', axis=1)

# Đảm bảo tất cả các cột trong X đều là số
X = X.astype(float)

print(f"Số lượng features: {X.shape[1]}")
print(f"Số lượng mẫu: {X.shape[0]}")

# 2. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# 3. Huấn luyện mô hình (Sử dụng Scikit-learn)
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
print("\nHuấn luyện mô hình bằng Scikit-learn thành công.")

# 4. Huấn luyện mô hình (Sử dụng Statsmodels để xem chi tiết)
# Thêm cột hệ số chặn (intercept) cho Statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

model_statsmodels = sm.OLS(y_train, X_train_sm).fit()
print("\n--- Tóm tắt Mô hình Hồi quy (Statsmodels) ---")
print(model_statsmodels.summary())
# Lưu ý:
# - R-squared: Tỷ lệ phương sai của biến phụ thuộc được giải thích bởi mô hình.
# - Adj. R-squared: R-squared đã điều chỉnh theo số lượng biến độc lập.
# - P>|t|: P-value cho từng hệ số. Nếu p-value < 0.05 (mức ý nghĩa phổ biến), biến đó có ý nghĩa thống kê.
# - Coef: Hệ số hồi quy cho từng biến.

# --- Đánh giá Mô hình ---
print("\n--- Đánh giá Mô hình trên Tập Kiểm tra ---")

# Dự đoán trên tập kiểm tra
y_pred_sklearn = model_sklearn.predict(X_test)
y_pred_statsmodels = model_statsmodels.predict(X_test_sm) # Sử dụng model Statsmodels

# Tính toán chỉ số đánh giá (sử dụng dự đoán từ Scikit-learn, kết quả tương tự với Statsmodels)
mse = mean_squared_error(y_test, y_pred_sklearn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_sklearn)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.4f}")

# Phân tích phần dư
residuals = y_test - y_pred_sklearn

plt.figure(figsize=(12, 5))

# Biểu đồ phân tán phần dư vs Giá trị dự đoán
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_pred_sklearn, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Giá trị Dự đoán (Predicted Price)')
plt.ylabel('Phần dư (Residuals)')
plt.title('Biểu đồ Phần dư vs Giá trị Dự đoán')

# Biểu đồ phân phối của phần dư
plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel('Phần dư (Residuals)')
plt.ylabel('Tần suất')
plt.title('Phân phối của Phần dư')

plt.tight_layout()
plt.show()

# Nhận xét về phần dư:
# - Biểu đồ phần dư vs dự đoán: Lý tưởng là các điểm phân bố ngẫu nhiên quanh đường 0, không có hình dạng rõ ràng (ví dụ: hình phễu). Nếu có hình dạng, có thể mô hình chưa bắt hết các mối quan hệ hoặc phương sai của sai số không đồng nhất.
# - Phân phối phần dư: Lý tưởng là gần giống phân phối chuẩn (hình chuông).

# --- Dự đoán và Diễn giải ---
print("\n--- Dự đoán và Diễn giải ---")

# Xem các hệ số hồi quy từ mô hình Scikit-learn
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_sklearn.coef_
})
# Thêm hệ số chặn
intercept_df = pd.DataFrame({'Feature': 'Intercept', 'Coefficient': model_sklearn.intercept_}, index=[0])
coefficients = pd.concat([intercept_df, coefficients], ignore_index=True)

print("\nHệ số hồi quy của mô hình:")
# Sắp xếp theo giá trị tuyệt đối để xem biến nào ảnh hưởng mạnh nhất
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
print(coefficients.sort_values(by='Abs_Coefficient', ascending=False).drop('Abs_Coefficient', axis=1))

# Diễn giải ví dụ một hệ số:
# - Nếu hệ số của 'RAM_GB' là 500,000: Giữ các yếu tố khác không đổi, việc tăng 1GB RAM
#   có liên quan đến việc tăng giá trung bình khoảng 500,000 VND.
# - Nếu hệ số của 'Shell_Type_Plastic' là -1,000,000 (so với nhóm cơ sở, ví dụ Metal):
#   Giữ các yếu tố khác không đổi, một chiếc laptop vỏ nhựa có giá trung bình thấp hơn
#   khoảng 1,000,000 VND so với chiếc có vỏ Metal (nhóm cơ sở của biến one-hot).

print("\n--- Hoàn thành ---")

# Có thể lưu mô hình nếu cần
# import joblib
# joblib.dump(model_sklearn, 'dell_price_linear_regression.pkl')
# print("Đã lưu mô hình vào file dell_price_linear_regression.pkl")