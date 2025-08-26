from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pickle
import joblib
import traceback
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import RF_inference_Regression as rf
import numpy as np
# Workaround: ensure PolynomialFeatures is importable for unpickling older sklearn objects
from sklearn.preprocessing import PolynomialFeatures
import sklearn.preprocessing._data  # ensure module path exists for pickle

# Ensure unpickling can find PolynomialFeatures at the older module path
try:
    if not hasattr(sklearn.preprocessing._data, "PolynomialFeatures"):
        sklearn.preprocessing._data.PolynomialFeatures = PolynomialFeatures
except Exception:
    # best-effort, continue so we can show clearer errors later
    pass

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
# 模型文件路径
DEFAULT_MODEL = os.path.join(BASE_DIR, "final_model.pkl")
DEFAULT_POLY = os.path.join(BASE_DIR, "poly_scaler.pkl")

# 固定参数名
CSV_COLUMNS = ["Glu", "FIB", "AST.ALT", "AG", "Age", "BMI", "NC", "Mallampati"]

# 默认值
DEFAULT_VALUES = ["4.22", "1.71", "0.74", "17.8", "21", "23.4", "43", "1"]

@app.route("/")
def index():
    # 把列名和默认值传给前端页面
    return render_template("index.html", columns=CSV_COLUMNS, defaults=DEFAULT_VALUES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("DEBUG: poly/model exist:", os.path.exists(DEFAULT_POLY), os.path.exists(DEFAULT_MODEL))

        # 接收前端传来的 CSV 格式文本
        data_text = request.form.get("data", "")
        print("DEBUG: Received raw data_text repr:", repr(data_text)[:1000])
        if not data_text or str(data_text).strip() == "":
            return jsonify({"error": "No input data provided"}), 400
        data_text = data_text.strip()

        # 转换成 DataFrame
        df = pd.read_csv(StringIO(data_text), header=None, encoding='utf-8')
        print("DEBUG: Parsed DataFrame shape:", df.shape)
        expected_cols = len(CSV_COLUMNS)
        if df.shape[1] != expected_cols:
            return jsonify({"error": f"Each row must have {expected_cols} columns, got {df.shape[1]}"}), 400

        # 填充缺失值
        try:
            df.fillna(df.median(numeric_only=True), inplace=True)
        except Exception:
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Label encode 对象型列
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df
        print("DEBUG: X dtypes:", X.dtypes.to_dict())

        # 加载 poly/scaler
        poly = scaler = None
        try:
            loaded = joblib.load(DEFAULT_POLY)
            if isinstance(loaded, tuple) and len(loaded) >= 2:
                poly, scaler = loaded[0], loaded[1]
            else:
                poly, scaler = loaded
            print("DEBUG: Loaded poly/scaler via joblib:", type(poly), type(scaler))
        except Exception as e_poly:
            print("WARN: joblib.load poly failed:", repr(e_poly))
            try:
                with open(DEFAULT_POLY, "rb") as pf:
                    poly, scaler = pickle.load(pf)
                print("DEBUG: Loaded poly/scaler via pickle:", type(poly), type(scaler))
            except Exception as e2:
                print("WARN: pickle.load poly failed, retry latin1:", repr(e2))
                with open(DEFAULT_POLY, "rb") as pf:
                    poly, scaler = pickle.load(pf, encoding='latin1')
                print("DEBUG: Loaded poly/scaler via pickle+latin1:", type(poly), type(scaler))

        X_scaled, _, _ = rf.preprocess_features(X, degree=2, scale=True, poly=poly, scaler=scaler)
        print("DEBUG: X_scaled shape/type:", getattr(X_scaled, "shape", None), type(X_scaled))

        # 加载模型
        model = None
        try:
            model = joblib.load(DEFAULT_MODEL)
            print("DEBUG: Loaded model via joblib:", type(model))
        except Exception as e_m1:
            print("WARN: joblib.load model failed:", repr(e_m1))
            try:
                with open(DEFAULT_MODEL, "rb") as mf:
                    model = pickle.load(mf)
                print("DEBUG: Loaded model via pickle:", type(model))
            except Exception as e_m2:
                print("WARN: pickle.load model failed, retry latin1:", repr(e_m2))
                with open(DEFAULT_MODEL, "rb") as mf:
                    model = pickle.load(mf, encoding='latin1')
                print("DEBUG: Loaded model via pickle+latin1:", type(model))

        if not hasattr(model, "predict"):
            raise RuntimeError("Loaded model has no predict method")

        print("DEBUG: Calling model.predict ...")
        y_pred = model.predict(X_scaled)
        print("DEBUG: Raw prediction type/shape:", type(y_pred), getattr(getattr(y_pred, 'shape', None), '__str__', lambda: '')())

        # 转成 Python list
        arr = np.asarray(y_pred).ravel()
        preds = [float(x) for x in arr.tolist()]
        print("DEBUG: preds sample:", preds[:10])

        # 新增：按阈值 0.75 计算标签和背景颜色
        threshold = 0.75
        results = []
        labels = []
        for v in preds:
            if v < threshold:
                label = "No or mild OSA"
                color = "#28a745"  # 绿色
            else:
                label = "Moderate to severe OSA"
                color = "#dc3545"  # 红色
            results.append({"value": float(v), "label": label, "color": color})
            labels.append(label)

        # ✅ 返回统一格式，predictions 字段改为直接返回 label（字符串），results 仍包含value/label/color
        return jsonify({"predictions": labels, "labels": labels, "results": results, "n_rows": int(len(preds))})

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR Traceback:\n", tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    app.run(debug=True)
