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
# 直接使用工作区根目录下的 final_model.pkl 与 poly_scaler.pkl
DEFAULT_MODEL = os.path.join(BASE_DIR, "final_model.pkl")
DEFAULT_POLY = os.path.join(BASE_DIR, "poly_scaler.pkl")

# 固定参数名（无需再读取 testing_data 的表头）
CSV_COLUMNS = ["Glu", "FIB", "AST.ALT", "AG", "Age", "BMI", "NC", "Mallampati"]

# 新增：默认值（与用户提供的一致）
DEFAULT_VALUES = ["4.22", "1.71", "0.74", "17.8", "21", "23.4", "43", "1"]

@app.route("/")
def index():
    # 直接传入固定列名与默认值
    return render_template("index.html", columns=CSV_COLUMNS, defaults=DEFAULT_VALUES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("DEBUG: poly/model exist:", os.path.exists(DEFAULT_POLY), os.path.exists(DEFAULT_MODEL))

        data_text = request.form.get("data", "")
        print("DEBUG: Received raw data_text repr:", repr(data_text)[:1000])
        if not data_text or str(data_text).strip() == "":
            return jsonify({"error": "No input data provided"}), 400
        data_text = data_text.strip()

        # parse CSV without header
        df = pd.read_csv(StringIO(data_text), header=None, encoding='utf-8')
        print("DEBUG: Parsed DataFrame shape:", df.shape)
        expected_cols = len(CSV_COLUMNS)
        if df.shape[1] != expected_cols:
            return jsonify({"error": f"Each row must have {expected_cols} columns, got {df.shape[1]}"}), 400

        # missing values
        try:
            df.fillna(df.median(numeric_only=True), inplace=True)
        except Exception:
            df = df.fillna(method='ffill').fillna(method='bfill')

        # label encode object cols
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        X = df
        print("DEBUG: X dtypes:", X.dtypes.to_dict())

        # load poly/scaler
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

        # load model
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

        arr = np.asarray(y_pred).ravel()
        preds = [float(x) for x in arr.tolist()]
        print("DEBUG: preds sample:", preds[:10])

        return jsonify({"predictions": preds, "n_rows": int(len(preds))})

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR Traceback:\n", tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    app.run(debug=True)
