import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
file_path = "c:/Users/Admin/OneDrive - ptit.edu.vn/tài liệu/AI/Học máy và ứng dụng/laptops.csv"
df = pd.read_csv(file_path)

# Chia tập dữ liệu thành train (80%) và test (20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Lưu tập train và test vào file CSV
train_df.to_csv("c:/Users/Admin/OneDrive - ptit.edu.vn/tài liệu/AI/Học máy và ứng dụng/laptops_train.csv", index=False)
test_df.to_csv("c:/Users/Admin/OneDrive - ptit.edu.vn/tài liệu/AI/Học máy và ứng dụng/laptops_test.csv", index=False)

print("Đã chia dữ liệu thành công và lưu vào file!")
