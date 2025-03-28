import pandas as pd
import joblib
import numpy as np

def predict_laptop_price():
    # 1. Tải mô hình và các bộ mã hóa
    model = joblib.load("laptop_price_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")

    # 2. Nhập dữ liệu từ người dùng
    print("📌 Nhập thông tin laptop để dự đoán giá:")
    ScreenSize = input("📏 Kích thước màn hình (e.g., 15.6 inch): ")
    CPU = input("⚡ CPU (e.g., Intel i5-1135G7): ")
    GPU = input("🎮 GPU (e.g., NVIDIA GTX 1650): ")
    RAM = input("🛠 RAM (e.g., 8GB): ")
    Shell = input("💻 Chất liệu vỏ (e.g., Aluminum, Plastic): ")
    Battery = input("🔋 Pin (e.g., 56Wh): ")
    Shop = input("🏬 Cửa hàng (e.g., Thegioididong, FPTShop): ")
    Storage = float(input("💾 Ổ cứng (GB): "))
    Weight = float(input("⚖ Trọng lượng (kg): "))

    # 3. Chuyển thành DataFrame
    new_data = pd.DataFrame([{
        "ScreenSize": ScreenSize,
        "CPU": CPU,
        "GPU": GPU,
        "RAM": RAM,
        "Shell": Shell,
        "Battery": Battery,
        "Shop": Shop,
        "Storage": Storage,
        "Weight": Weight
    }])

    # 4. Xử lý dữ liệu như khi huấn luyện
    categorical_features = ["ScreenSize", "CPU", "GPU", "RAM", "Shell", "Battery", "Shop"]
    
    # Mã hóa các cột text
    encoded_features = encoder.transform(new_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # Ghép dữ liệu đã mã hóa với các cột số
    df_numeric = new_data.drop(columns=categorical_features)
    df_final = pd.concat([df_numeric, encoded_df], axis=1)

    # 5. Chuẩn hóa dữ liệu
    X_scaled = scaler.transform(df_final)

    # 6. Dự đoán giá laptop
    predicted_price = model.predict(X_scaled)[0]

    # 7. Hiển thị kết quả
    print(f"\n💰 Giá laptop dự đoán: {predicted_price:,.0f} VNĐ")

# Gọi hàm để dự đoán
predict_laptop_price()
