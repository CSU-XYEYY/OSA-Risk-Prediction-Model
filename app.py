from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import pickle
import joblib
import traceback
from io import StringIO
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
# 模型文件路径 - 使用原始的 XGBoost 分类模型
DEFAULT_MODEL = os.path.join(BASE_DIR, "xgboost_classifier_final.pkl")

# 固定参数名 - 与 inference.py 中的特征名一致
CSV_COLUMNS = ["Glu(mmol/L)", "FIB(mg/dL)", "AST/ALT", "AG(mmol/L)", "Age", "BMI(kg/m^2)", "NC(cm)", "Mallampati"]

# 默认值
DEFAULT_VALUES = ["4.22", "1.71", "0.74", "17.8", "21", "23.4", "43", "1"]

@app.route("/")
def index():
    # 把列名和默认值传给前端页面
    return render_template("index.html", columns=CSV_COLUMNS, defaults=DEFAULT_VALUES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("DEBUG: model exists:", os.path.exists(DEFAULT_MODEL))

        # 接收前端传来的数据
        # 兼容两种格式：
        # 1. 表单数据：form field 'data' 包含 CSV 字符串
        # 2. JSON 数据：{ "data": [[value1, value2, ...], ...] }
        print("DEBUG: Content-Type:", request.content_type)
        
        df = None
        if request.content_type and 'application/json' in request.content_type:
            # 处理 JSON 格式数据
            json_body = request.get_json(silent=True)
            if json_body and isinstance(json_body, dict) and 'data' in json_body:
                data_list = json_body['data']
                if isinstance(data_list, list) and len(data_list) > 0:
                    # 将列表转换为 DataFrame
                    df = pd.DataFrame(data_list)
                    print("DEBUG: Parsed DataFrame from JSON, shape:", df.shape)
                else:
                    return jsonify({"error": "Invalid JSON data format"}), 400
            else:
                return jsonify({"error": "Invalid JSON request"}), 400
        else:
            # 处理表单数据（CSV 字符串）
            data_text = ""
            if 'data' in request.form:
                data_text = request.form.get("data", "")
            else:
                # 尝试原始 body（纯文本 CSV）
                data_text = request.get_data(as_text=True) or ""
            
            print("DEBUG: Received raw data_text repr:", repr(data_text)[:1000])
            if not data_text or str(data_text).strip() == "":
                return jsonify({"error": "No input data provided"}), 400
            
            data_text = data_text.strip()
            # 转换成 DataFrame
            df = pd.read_csv(StringIO(data_text), header=None, encoding='utf-8')
            print("DEBUG: Parsed DataFrame from CSV, shape:", df.shape)
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
        
        # 重命名列名以匹配模型期望的特征名
        X.columns = ['Glu', 'FIB', 'AST. ALT', 'AG', 'Age', 'BMI', 'NC', 'Mallampati']
        print("DEBUG: X columns after rename:", list(X.columns))

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
            
        # 修复XGBoost模型兼容性问题
        if hasattr(model, '__class__') and 'XGBClassifier' in str(model.__class__):
            print("DEBUG: Detected XGBoost classifier, checking for compatibility issues...")
            # 设置缺失的属性以避免get_params()错误
            try:
                if not hasattr(model, 'use_label_encoder'):
                    print("DEBUG: Setting use_label_encoder to False")
                    model.use_label_encoder = False
            except Exception as e:
                print("DEBUG: Error setting use_label_encoder:", e)
                
            try:
                if not hasattr(model, 'gpu_id'):
                    print("DEBUG: Setting gpu_id to -1")
                    model.gpu_id = -1
            except Exception as e:
                print("DEBUG: Error setting gpu_id:", e)
                
            try:
                if not hasattr(model, 'eval_metric'):
                    print("DEBUG: Setting eval_metric to 'logloss'")
                    model.eval_metric = 'logloss'
            except Exception as e:
                print("DEBUG: Error setting eval_metric:", e)
                
            # 设置标签编码器属性
            try:
                if not hasattr(model, '_le'):
                    print("DEBUG: Setting _le attribute")
                    from sklearn.preprocessing import LabelEncoder
                    model._le = LabelEncoder()
                    model._le.classes_ = np.array([0, 1])
            except Exception as e:
                print("DEBUG: Error setting _le attribute:", e)

        print("DEBUG: Calling model.predict ...")
        try:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)
        except AttributeError as e:
            print(f"DEBUG: Prediction error: {e}")
            print("DEBUG: Trying alternative prediction method...")
            # 尝试直接使用底层booster进行预测
            try:
                import xgboost as xgb
                # 获取底层booster
                booster = model.get_booster()
                # 将数据转换为DMatrix，设置特征名
                dmatrix = xgb.DMatrix(X.values, feature_names=list(X.columns))
                # 获取原始预测
                raw_pred = booster.predict(dmatrix)
                # 对于分类器，需要将原始预测转换为概率
                if hasattr(model, 'objective') and 'binary' in str(model.objective):
                    # 二分类：使用sigmoid转换
                    prob = 1 / (1 + np.exp(-raw_pred))
                    y_prob = np.column_stack([1 - prob, prob])
                    y_pred = (prob > 0.5).astype(int)
                else:
                    # 多分类或其他
                    y_pred = raw_pred
                    y_prob = raw_pred
                print("DEBUG: Used alternative prediction method")
            except Exception as e2:
                print(f"DEBUG: Alternative method failed: {e2}")
                raise e
        print("DEBUG: Raw prediction type/shape:", type(y_pred), getattr(y_pred, 'shape', 'no shape'))
        print("DEBUG: Probability shape:", getattr(y_prob, 'shape', 'no shape'))

        # 转成 Python list
        pred_classes = [int(x) for x in y_pred.tolist()]
        pred_probs_class1 = [float(x) for x in y_prob[:, 1].tolist()]
        print("DEBUG: pred_classes sample:", pred_classes[:10])
        print("DEBUG: pred_probs_class1 sample:", pred_probs_class1[:10])

        # 生成结果
        results = []
        labels = []
        for i, (cls, prob) in enumerate(zip(pred_classes, pred_probs_class1)):
            if cls == 0:
                label = "No or mild OSA"
                color = "#28a745"  # 绿色
            else:
                label = "Moderate to severe OSA"
                color = "#dc3545"  # 红色
            
            # 添加置信度信息
            confidence = prob if cls == 1 else (1 - prob)
            results.append({
                "class": int(cls),
                "probability_class_1": float(prob),
                "confidence": float(confidence),
                "label": label,
                "color": color
            })
            labels.append(label)

        # ✅ 返回统一格式
        return jsonify({
            "predictions": pred_classes,
            "probabilities": pred_probs_class1,
            "labels": labels,
            "results": results,
            "n_rows": int(len(pred_classes))
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("ERROR Traceback:\n", tb)
        return jsonify({"error": str(e), "traceback": tb}), 500

if __name__ == "__main__":
    app.run(debug=True)