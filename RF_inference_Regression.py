# filepath: web_app_regression/web_app_regression/src/RF_inference_Regression.py
# -*- coding:utf-8 -*-
'''
回归推理的主要逻辑
'''
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder

# 加载和预处理数据
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1, encoding='ISO-8859-1')
    data.fillna(data.median(), inplace=True)  # 填充缺失值
    object_columns = data.select_dtypes(include=['object']).columns
    encoded_data = data.copy()
    for col in object_columns:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(encoded_data[col])
    return encoded_data

# 特征工程：生成多项式特征并标准化
def preprocess_features(X, degree, scale=True, poly=None, scaler=None):
    poly = poly or PolynomialFeatures(degree)
    X_poly = poly.transform(X)
    if scale:
        scaler = scaler or StandardScaler()
        X_scaled = scaler.transform(X_poly)
        return X_scaled, poly, scaler
    else:
        return X_poly, poly, None

# 加载训练好的模型
def load_trained_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# 进行预测
def predict(file_path, model_filename, degree=2, scale=True):
    # 加载数据并进行预处理
    encoded_data = load_and_preprocess_data(file_path)
    X = encoded_data.iloc[:, 0:-1]  # 特征列
    y = encoded_data.iloc[:, -1]    # 标签列

    # 加载训练时保存的多项式转换器和标准化器
    with open('models/poly_scaler.pkl', 'rb') as file:
        poly, scaler = pickle.load(file)

    # 特征处理
    X_scaled, _, _ = preprocess_features(X, degree, scale, poly=poly, scaler=scaler)

    # 加载训练好的回归模型
    model = load_trained_model(model_filename)

    # 进行预测
    y_pred = model.predict(X_scaled)

    return y_pred, y

# 生成结果 DataFrame
def generate_results_df(encoded_data, y_pred):
    results_df = pd.DataFrame({
        'Index': encoded_data.index,
        'Predicted': y_pred,
        'Actual': encoded_data.iloc[:, -1]
    })
    return results_df

# 主推理函数
def main(file_path, model_filename='models/model_fold_5.pkl', degree=2, scale=True):
    y_pred, y_actual = predict(file_path, model_filename, degree, scale)
    encoded_data = load_and_preprocess_data(file_path)
    results_df = generate_results_df(encoded_data, y_pred)
    return results_df

if __name__ == "__main__":
    results = main('../testing_data_20250704_142241.csv', model_filename='models/model_fold_5.pkl', degree=2, scale=True)
    print(results.head())