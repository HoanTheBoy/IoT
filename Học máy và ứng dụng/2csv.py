import pandas as pd

excel_file = 'data.xlsx'  # Đường dẫn đến file Excel
csv_file = 'data.csv'  # Đường dẫn đến file CSV

def convert_excel_to_csv(excel_file, csv_file):
    df = pd.read_excel(excel_file)
    df.to_csv(csv_file, index=False)
    print(f"Đã chuyển đổi {excel_file} sang {csv_file}.")

def fill_null(csv_file):
    df = pd.read_csv(csv_file)

    df.fillna(df.mean(numeric_only=True), inplace=True)

    for col in df.select_dtypes(include=["object"]).columns: 
        most_frequent_value = df[col].mode()[0]
        df[col].fillna(most_frequent_value, inplace=True)

    df.to_csv(csv_file, index=False)
    print("Đã điền giá trị null và lưu lại vào file CSV.")

def convert_price(price):
    if isinstance(price, str):
        price = price.replace("₫", "").replace(",", "").strip()
        return float(price) if price else None
    return price  

def func_convert_price(csv_file):
    df = pd.read_csv(csv_file)
    df["Price"] = df["Price"].apply(convert_price)
    df.to_csv(csv_file, index=False)

    print("Đã chuyển đổi định dạng giá và lưu lại vào file CSV.")

while(1):
    choise = input("Nhập 1 để chuyển đổi file excel sang csv, 2 để điền giá trị null, 3 để định dạng giá, 0 để thoát: ")
    if choise == '1':
        convert_excel_to_csv(excel_file, csv_file)
    elif choise == '2':
        fill_null(csv_file)
    elif choise == '3':
        func_convert_price(csv_file)
    elif choise == '0':
        break
    else:
        print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
