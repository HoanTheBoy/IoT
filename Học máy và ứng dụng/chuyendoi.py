import pandas as pd

def replace_nhua_in_csv(input_file, output_file):
    # Đọc file CSV
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # Thay thế từ "nhựa" thành "plastic" trong toàn bộ DataFrame
    df = df.applymap(lambda x: x.replace("nhựa", "plastic") if isinstance(x, str) else x)
    
    # Ghi lại file CSV mới
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Đã thay thế 'nhựa' thành 'plastic' và lưu vào {output_file}")

# Ví dụ sử dụng
replace_nhua_in_csv("data.csv,new_data.csv")
