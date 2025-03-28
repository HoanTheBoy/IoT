import pandas as pd
import numpy as np
import joblib  # Dùng để lưu và tải mô hình
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Đọc dữ liệu
df = pd.read_csv("datafake.csv")

y = df["Price"]

# 3. Xử lý dữ liệu - Mã hóa các cột dạng text
categorical_features = ["ScreenSize", "CPU", "GPU", "RAM", "Shell", "Battery", "Shop"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# 4. Ghép dữ liệu đã mã hóa với các cột số
df_numeric = df.drop(columns=categorical_features + ["Name", "Price"])
df_final = pd.concat([df_numeric, encoded_df], axis=1)

# Kiểm tra lại số dòng của X và y
print(f"🛠 Số dòng của X: {df_final.shape[0]}")
print(f"🛠 Số dòng của y: {y.shape[0]}")

# 5. Chia tập dữ liệu train/test
X = df_final

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Lưu mô hình, encoder, và scaler vào file
joblib.dump(model, "laptop_price_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Mô hình đã được lưu thành công!")

# 8. Đánh giá mô hình
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"🎯 R^2 Score: {r2:.4f}")
