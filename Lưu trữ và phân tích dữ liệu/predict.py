import joblib
import pandas as pd
import seaborn as sns

features_to_normalize = ['Past_Purchase_Trends' ,'Price' ,'Discount' ,'Competitor_Price']

def predict_sales(data: pd.DataFrame):
    # Load mô hình và bộ xử lý dữ liệu
    model = joblib.load('catboost_model.pkl')
    sc = joblib.load('scaler.pkl')
    
    data = data.copy()
    data = data.drop('Sales_Quantity', axis = 1)
    data['Date'] = pd.to_datetime(data['Date'])

    data.isna().sum()
    data = data.dropna()
    data['month'] = data['Date'].dt.month
    sns.histplot(data['Price'])
    sns.histplot(data['Competitor_Price'])
    sns.histplot(data['Discount'])
    data['Quarter'] = data['Date'].dt.quarter
    data['Price_Difference'] = data['Price'] - data['Competitor_Price']
    data[features_to_normalize] = sc.transform(data[features_to_normalize])
    data = data.drop('Date', axis = 1)
    
    # Dự đoán
    predictions = model.predict(data)
    return pd.Series(predictions)

test_data = pd.read_csv('./Test.csv')
predictions = predict_sales(test_data)
print(predictions)
predictions.to_csv('predictions.csv', index=False)
