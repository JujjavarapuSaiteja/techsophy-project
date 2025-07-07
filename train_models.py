import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
os.makedirs("models", exist_ok=True)
#taking only dataframes,for more values we can take dataset and read the csv files using pandas
diabetes_data = pd.DataFrame({
    'Age': [25, 45, 60, 30, 50, 35, 40, 55, 29],
    'BMI': [22, 28, 33, 25, 31, 30, 27, 34, 23],
    'Glucose': [85, 140, 160, 95, 145, 135, 120, 150, 88],
    'BloodPressure': [70, 90, 100, 80, 95, 85, 88, 92, 76],
    'Outcome': [0, 1, 1, 0, 1, 1, 0, 1, 0]
})

X_dia = diabetes_data[['Age', 'BMI', 'Glucose', 'BloodPressure']]
y_dia = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_dia, y_dia, test_size=0.2)

diabetes_model = LogisticRegression()
diabetes_model.fit(X_train, y_train)
joblib.dump(diabetes_model, 'models/diabetes_model.pkl')

heart_data = pd.DataFrame({
    'Age': [30, 60, 55, 40, 45, 70, 65, 50, 48],
    'BloodPressure': [120, 160, 150, 130, 135, 170, 165, 140, 138],
    'Cholesterol': [180, 260, 250, 200, 210, 270, 260, 240, 230],
    'Smoker': [0, 1, 1, 0, 1, 1, 1, 1, 0],
    'Outcome': [0, 1, 1, 0, 1, 1, 1, 1, 0]
})

X_heart = heart_data[['Age', 'BloodPressure', 'Cholesterol', 'Smoker']]
y_heart = heart_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.2)

heart_model = DecisionTreeClassifier()
heart_model.fit(X_train, y_train)
joblib.dump(heart_model, 'models/heart_model.pkl')
obesity_data = pd.DataFrame({
    'BMI': [18, 22, 30, 35, 25, 28, 33, 36, 24],
    'ActivityLevel': [0, 0, 2, 2, 1, 1, 2, 2, 1],  # 0=high, 1=moderate, 2=low
    'Outcome': [0, 0, 1, 1, 0, 0, 1, 1, 0]
})

X_obesity = obesity_data[['BMI', 'ActivityLevel']]
y_obesity = obesity_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X_obesity, y_obesity, test_size=0.2)

obesity_model = LogisticRegression()
obesity_model.fit(X_train, y_train)
joblib.dump(obesity_model, 'models/obesity_model.pkl')

print("All models trained and saved to 'models/' folder")
