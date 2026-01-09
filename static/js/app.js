document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    const table = document.getElementById("data-table").getElementsByTagName("tbody")[0];

    const resultContainer = document.getElementById("prediction-container");
    const badge = document.getElementById("prediction-badge");
    const sub = document.getElementById("prediction-sub");
    const probInfo = document.getElementById("probability-info");

    const predList = document.getElementById("pred-list"); // 用于显示多行结果

    // 提交表单
    form.addEventListener("submit", (e) => {
        e.preventDefault();

        // 收集表格数据
        const data = [];
        for (let r = 0; r < table.rows.length; r++) {
            const row = [];
            const inputs = table.rows[r].querySelectorAll("input");
            inputs.forEach(input => {
                row.push(parseFloat(input.value) || 0);
            });
            data.push(row);
        }

        // 调用后端 API
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: data })
        })
        .then(response => response.json())
        .then(result => {
            if (result.predictions) {
                showPredictions(result);
            } else {
                showError("No predictions returned");
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
            showError("Error: could not get prediction");
        });
    });

    // 显示预测结果
    function showPredictions(result) {
        const predictions = result.predictions || [];
        const probabilities = result.probabilities || [];
        const labels = result.labels || [];
        const resultsData = result.results || [];

        if (predictions.length === 0) {
            showError("No predictions available");
            return;
        }

        // 显示第一个结果
        const firstResult = resultsData[0];
        badge.textContent = firstResult.label;
        badge.style.background = firstResult.color;
        sub.textContent = `Class: ${firstResult.class === 0 ? '0 (No/mild OSA)' : '1 (Moderate/severe OSA)'}`;
        
        // 隐藏概率信息
        probInfo.innerHTML = "";

        // 显示所有结果在 pred-list
        predList.innerHTML = ""; // 清空旧内容
        resultsData.forEach((res, idx) => {
            const p = document.createElement("p");
            p.textContent = `Row ${idx + 1}: Class ${res.class} (${res.label})`;
            p.style.color = res.color;
            p.style.margin = "4px 0";
            p.style.padding = "4px 8px";
            p.style.backgroundColor = `${res.color}15`;
            p.style.borderRadius = "4px";
            predList.appendChild(p);
        });

        resultContainer.style.display = "block";
    }

    // 错误显示
    function showError(msg) {
        badge.textContent = "--";
        badge.style.background = "#6c757d";
        sub.textContent = msg;
        sub.style.color = "gray";
        probInfo.innerHTML = "";
        predList.innerHTML = "";
        resultContainer.style.display = "block";
    }
});
