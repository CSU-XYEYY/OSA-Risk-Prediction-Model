document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    const table = document.getElementById("data-table").getElementsByTagName("tbody")[0];
    const addRowBtn = document.getElementById("add-row");
    const removeRowBtn = document.getElementById("remove-row");

    const resultContainer = document.getElementById("prediction-container");
    const badge = document.getElementById("prediction-badge");
    const sub = document.getElementById("prediction-sub");

    const predList = document.getElementById("pred-list"); // 用于显示多行结果

    // 添加一行
    addRowBtn.addEventListener("click", () => {
        const newRow = table.rows[0].cloneNode(true);
        Array.from(newRow.querySelectorAll("input")).forEach(input => input.value = "");
        table.appendChild(newRow);
    });

    // 删除最后一行
    removeRowBtn.addEventListener("click", () => {
        if (table.rows.length > 1) {
            table.deleteRow(table.rows.length - 1);
        }
    });

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
                showPredictions(result.predictions);
            } else {
                showError("No predictions returned");
            }
        })
        .catch(err => {
            console.error("Prediction error:", err);
            showError("Error: could not get prediction");
        });
    });

    // 显示多行预测结果
    function showPredictions(pred_values) {
        const threshold = 0.75;

        // 显示第一个结果在 badge
        const firstPred = pred_values[0];
        badge.textContent = firstPred.toFixed(3);
        sub.textContent = firstPred < threshold ? "no or mild OSA" : "moderate to severe OSA";
        sub.style.color = firstPred < threshold ? "green" : "red";

        // 显示所有结果在 pred-list
        predList.innerHTML = ""; // 清空旧内容
        pred_values.forEach((val, idx) => {
            const p = document.createElement("p");
            p.textContent = `Row ${idx + 1}: ${val.toFixed(3)} → ${val < threshold ? "no or mild OSA" : "moderate to severe OSA"}`;
            p.style.color = val < threshold ? "green" : "red";
            predList.appendChild(p);
        });

        resultContainer.style.display = "block";
    }

    // 错误显示
    function showError(msg) {
        badge.textContent = "--";
        sub.textContent = msg;
        sub.style.color = "gray";
        predList.innerHTML = "";
        resultContainer.style.display = "block";
    }
});
