import pandas as pd
import numpy as np
import joblib  # Dùng để lưu và tải mô hình
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. Đọc dữ liệu
df = pd.read_csv("laptops.csv")

y = df["Final Price"]

df["GPU"].fillna("No GPU", inplace=True)
# 3. Xử lý dữ liệu - Mã hóa các cột dạng text
categorical_features = ["Status", "Brand", "Model", "CPU", "GPU", "Storage type", "Screen", "Touch"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# 4. Ghép dữ liệu đã mã hóa với các cột số
df_numeric = df.drop(columns=categorical_features + ["Laptop", "RAM", "Storage", "Final Price"])
df_final = pd.concat([df_numeric, encoded_df], axis=1)

# 5. Chia tập dữ liệu train/test
X = df_final

# Vẽ ma trận tương quan
numeric_columns = ['RAM', 'Storage', 'Final Price']
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan giữa các đặc trưng số")
plt.show()

# Vẽ biểu đồ tần suất cho các đặc trưng phân loại
categorical_columns = ['Brand', 'CPU']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(categorical_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
    axes[i].set_title(f"Phân bố của {col}")
    axes[i].set_ylabel("Count")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Lưu mô hình, encoder, và scaler vào file
joblib.dump(model, "laptop_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("##  Mô hình đã được lưu thành công!")

# 8. Đánh giá mô hình
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"##  R² score: {r2:.4f}")
# Hiển thị một số giá trị thực tế so với dự đoán
comparison_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(comparison_df.sample(10))  # Hiển thị ngẫu nhiên 10 dòng để kiểm tra
