# 项目环境与运行说明（更新）

以下为根据 app.py 运行所需的依赖和运行说明，包含已更新的 requirements.txt 建议依赖及精简 README 指南。

---

## 推荐 Python 版本
- Python 3.8 ~ 3.10（部分旧的 pickle/模块路径在新版本 sklearn 中可能不兼容，建议使用 3.8 或 3.9 以减少不兼容风险）

---

## requirements.txt（建议内容）
将下面内容保存为 requirements.txt，或直接替换原文件：

Flask==2.1.2
pandas==1.4.2
numpy==1.22.3
scikit-learn==1.0.2
joblib==1.1.0
matplotlib==3.5.1
openpyxl==3.0.9
Werkzeug==2.1.2

说明：
- 明确加入 joblib，可避免部分环境下 joblib 从 sklearn 包外单独使用时报错。
- 固定 scikit-learn==1.0.2 与 app.py 中对旧模块路径的处理更兼容。

---

## 简短 README （README.md 建议替换内容）
1. 克隆或拷贝代码到本地目录。
2. 创建虚拟环境并激活（示例）：
   - python -m venv .venv
   - Windows: .venv\Scripts\activate
   - macOS/Linux: source .venv/bin/activate
3. 安装依赖：
   - pip install --upgrade pip
   - pip install -r requirements.txt
4. 必要文件：
   - 确保项目根目录包含模型文件：final_model.pkl 和 poly_scaler.pkl（app.py 会加载）。
   - 确保 templates/index.html 存在（Flask 前端页面）。
5. 启动服务：
   - python app.py
   - 默认在 http://127.0.0.1:5000/ 可访问。
6. 常见问题与排查：
   - Pickle 反序列化错误：若模型/预处理器由不同 sklearn 版本保存，app.py 已包含对 pickle/joblib 与 latin1 编码的多重尝试；推荐使用 scikit-learn==1.0.2 与保存模型时一致的环境重新导出模型。
   - 若遇到 "PolynomialFeatures" 找不到问题，app.py 已做兼容性处理，但仍建议在相同 sklearn 版本下导出 poly 对象。
   - 若收到列数不对或 CSV 解析错误，请确认上传的 CSV 无表头且列数与代码中 CSV_COLUMNS 一致。
7. 其他：
   - 仅作本地/受控环境测试，请勿直接在不受信任环境暴露生产模型文件。
