from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Đọc dữ liệu và chuẩn bị mô hình KNN
# (Lưu ý: Bạn cần thay đổi đường dẫn đến file dữ liệu của bạn)
df = pd.read_csv('Health_insurance.csv')
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận giá trị từ form
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])



    # Chuẩn hóa dữ liệu
    input_data = np.array([[age, 
                            1 if sex == 1 else 0, 
                            1 if sex == 0 else 0, 
                            bmi, 
                            children, 
                            1 if smoker == 0 else 0, 
                            1 if smoker == 1 else 0, 
                            1 if region == 1 else 0, 
                            1 if region == 2 else 0, 
                            1 if region == 3 else 0, 
                            1 if region == 4 else 0
                            ]])
    input_data_scaled = scaler.transform(input_data)

    # Dự đoán chi phí
    prediction = knn_model.predict(input_data_scaled)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
