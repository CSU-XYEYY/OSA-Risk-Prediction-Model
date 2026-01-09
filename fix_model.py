import pickle
import joblib
import xgboost as xgb

def fix_model_file(model_path):
    """修复XGBoost模型文件，移除use_label_encoder参数"""
    try:
        # 尝试用joblib加载
        model = joblib.load(model_path)
        print(f"Loaded model with joblib: {type(model)}")
    except Exception as e:
        print(f"Joblib load failed: {e}")
        try:
            # 尝试用pickle加载
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded model with pickle: {type(model)}")
        except Exception as e2:
            print(f"Pickle load failed: {e2}")
            try:
                # 尝试用latin1编码加载
                with open(model_path, 'rb') as f:
                    model = pickle.load(f, encoding='latin1')
                print(f"Loaded model with pickle+latin1: {type(model)}")
            except Exception as e3:
                print(f"All load methods failed: {e3}")
                return None
    
    # 检查是否是XGBoost模型
    if hasattr(model, '__class__') and 'XGBClassifier' in str(model.__class__):
        print("Detected XGBoost classifier")
        
        # 尝试移除use_label_encoder属性
        try:
            if hasattr(model, 'use_label_encoder'):
                print(f"Removing use_label_encoder attribute (current value: {model.use_label_encoder})")
                del model.use_label_encoder
                print("use_label_encoder attribute removed")
        except Exception as e:
            print(f"Error removing use_label_encoder: {e}")
        
        # 保存修复后的模型
        fixed_path = model_path.replace('.pkl', '_fixed.pkl')
        try:
            joblib.dump(model, fixed_path)
            print(f"Fixed model saved to: {fixed_path}")
            return fixed_path
        except Exception as e:
            print(f"Error saving fixed model: {e}")
            return None
    else:
        print("Model is not an XGBoost classifier")
        return None

if __name__ == "__main__":
    model_path = "xgboost_classifier_final.pkl"
    fixed_path = fix_model_file(model_path)
    
    if fixed_path:
        print(f"\nModel fixed successfully!")
        print(f"Original: {model_path}")
        print(f"Fixed: {fixed_path}")
        
        # 测试加载修复后的模型
        try:
            model = joblib.load(fixed_path)
            print(f"\nTest loading fixed model: Success!")
            print(f"Model type: {type(model)}")
            
            # 测试预测
            import numpy as np
            test_data = np.array([[4.22, 1.71, 0.74, 17.8, 21, 23.4, 43, 1]])
            print(f"\nTest prediction shape: {test_data.shape}")
            
            try:
                pred = model.predict(test_data)
                proba = model.predict_proba(test_data)
                print(f"Prediction: {pred}")
                print(f"Probabilities: {proba}")
                print("Model works correctly!")
            except Exception as e:
                print(f"Prediction error: {e}")
                
        except Exception as e:
            print(f"Error loading fixed model: {e}")
    else:
        print("Failed to fix model")