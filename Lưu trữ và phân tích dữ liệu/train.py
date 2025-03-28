
import pandas as pd 
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('./Train.csv')
test_df = pd.read_csv('./Test.csv')
# sub_df = pd.read_csv('./Submission.csv')

train_df.isna().sum()
train_df = train_df.dropna()

train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df['Category'].unique()
train_df['Brand'].unique()
train_df['month'] = train_df['Date'].dt.month

sns.histplot(train_df['Competitor_Price'])
sns.histplot(train_df['Discount'])
sns.histplot(train_df['Price'])
sns.histplot(train_df['Sales_Quantity'])

train_df['Quarter'] = train_df['Date'].dt.quarter
train_df['Price_Difference'] = train_df['Price'] - train_df['Competitor_Price']

features_to_normalize = ['Past_Purchase_Trends' ,'Price' ,'Discount' ,'Competitor_Price']

sc = StandardScaler()

train_df[features_to_normalize] = sc.fit_transform(train_df[features_to_normalize])
sns.lineplot(x = train_df['Date'], y = train_df['Sales_Quantity'])
train_df = train_df.drop('Date', axis = 1)

test_df = test_df.drop('Sales_Quantity', axis = 1)
test_df['Date'] = pd.to_datetime(test_df['Date'])

test_df.isna().sum()
test_df = test_df.dropna()
test_df['month'] = test_df['Date'].dt.month
sns.histplot(test_df['Price'])
sns.histplot(test_df['Competitor_Price'])
sns.histplot(test_df['Discount'])
test_df['Quarter'] = test_df['Date'].dt.quarter
test_df['Price_Difference'] = test_df['Price'] - test_df['Competitor_Price']
test_df[features_to_normalize] = sc.transform(test_df[features_to_normalize])
test_df = test_df.drop('Date', axis = 1)
X, y = train_df.drop('Sales_Quantity', axis = 1), train_df['Sales_Quantity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
categorical_cols = ['Brand', 'Category']
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.001,
    cat_features = categorical_cols,
    loss_function='RMSE',
    task_type="GPU",  
    devices='0'  
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, early_stopping_rounds=500)

import joblib
joblib.dump(model, 'catboost_model.pkl')
joblib.dump(sc, 'scaler.pkl')
joblib.dump(categorical_cols, 'categorical_cols.pkl')

# predictions = model.predict(test_df)
# sub_df['Sales_Quantity'] = predictions
# sns.histplot(sub_df['Sales_Quantity'])
# sub_df.to_csv('submission.csv', index = False)