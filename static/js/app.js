document.addEventListener('DOMContentLoaded', function () {
    // Ensure result nodes exist
    let resultDiv = document.getElementById('result');
    if (!resultDiv) {
        resultDiv = document.createElement('div');
        resultDiv.id = 'result';
        resultDiv.style.display = 'none';
        document.body.appendChild(resultDiv);
    }
    let summaryEl = document.getElementById('summary');
    if (!summaryEl) {
        summaryEl = document.createElement('div');
        summaryEl.id = 'summary';
        resultDiv.appendChild(summaryEl);
    }
    let predList = document.getElementById('pred-list');
    if (!predList) {
        predList = document.createElement('div');
        predList.id = 'pred-list';
        resultDiv.appendChild(predList);
    }

    // Table row controls (if present)
    const table = document.getElementById('data-table');
    const addRowBtn = document.getElementById('add-row');
    const removeRowBtn = document.getElementById('remove-row');

    if (addRowBtn && table) {
        addRowBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const tbody = table.querySelector('tbody');
            if (!tbody || tbody.rows.length === 0) return;
            const newRow = tbody.rows[0].cloneNode(true);
            Array.from(newRow.querySelectorAll('input.cell-input')).forEach(inp => inp.value = '');
            tbody.appendChild(newRow);
        });
    }

    if (removeRowBtn && table) {
        removeRowBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const tbody = table.querySelector('tbody');
            if (tbody && tbody.rows.length > 1) tbody.deleteRow(-1);
        });
    }

    // Choose form: prefer predict-form (table), otherwise regression-form
    const form = document.getElementById('predict-form') || document.getElementById('regression-form');
    if (!form) {
        console.warn('No predict-form or regression-form found in DOM.');
        return;
    }

    function showPredictionResult(resJson) {
        const container = document.getElementById('prediction-container') || document.getElementById('result');
        if (!container) return;

        // Resolve label from common fields
        let label = resJson.predicted_label || resJson.label || resJson.predicted || (Array.isArray(resJson.predictions) ? resJson.predictions[0] : undefined);
        // Map numeric/string indicators to human label
        if (label === 1 || label === '1' || String(label).toLowerCase() === 'positive') label = 'CRC';
        if (label === 0 || label === '0' || String(label).toLowerCase() === 'negative') label = 'Normal';
        label = label ?? 'Unknown';

        const cls = (String(label).toLowerCase().includes('crc') || String(label).toLowerCase().includes('positive') || label === '1') ? 'crc' : 'normal';

        const badge = document.getElementById('prediction-badge');
        if (badge) {
            badge.textContent = label;
            badge.classList.remove('crc', 'normal');
            badge.classList.add(cls);
        }

        // Summary and predictions list (for table mode)
        const nRows = resJson.n_rows ?? (Array.isArray(resJson.predictions) ? resJson.predictions.length : undefined);
        summaryEl.textContent = nRows ? `Predicted ${nRows} rows` : '';

        if (Array.isArray(resJson.predictions)) {
            predList.innerHTML = '<h3>Top predictions</h3>';
            const ul = document.createElement('ul');
            resJson.predictions.slice(0, 20).forEach((p, i) => {
                const li = document.createElement('li');
                li.textContent = `${i}: ${p}`;
                ul.appendChild(li);
            });
            predList.appendChild(ul);
        }

        // Additional details
        const sub = document.getElementById('prediction-sub');
        if (sub) {
            const details = [];
            if (resJson.score !== undefined) details.push(`Score: ${resJson.score}`);
            if (resJson.probability !== undefined) details.push(`Probability: ${resJson.probability}`);
            sub.textContent = details.join(' • ');
        }

        container.style.display = 'block';
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // If predict-form: collect table rows into CSV and upload as FormData('data')
        if (form.id === 'predict-form') {
            const tableEl = document.getElementById('data-table');
            let csvText = '';
            if (tableEl) {
                const rows = tableEl.querySelectorAll('tbody tr');
                const rowVals = [];
                rows.forEach(r => {
                    const vals = Array.from(r.querySelectorAll('input.cell-input')).map(i => {
                        const v = (i.value ?? '').trim();
                        return v.includes(',') ? `"${v.replace(/"/g, '""')}"` : v;
                    });
                    rowVals.push(vals.join(','));
                });
                csvText = rowVals.join('\n');
            } else {
                const ta = document.getElementById('csv-input');
                csvText = ta ? ta.value.trim() : '';
            }

            if (!csvText) {
                alert('No input data found to predict.');
                return;
            }

            const fd = new FormData();
            fd.append('data', csvText);

            try {
                const resp = await fetch('/predict', { method: 'POST', body: fd });
                const resJson = await resp.json();
                if (!resp.ok) throw new Error(resJson.error || 'Prediction failed');
                showPredictionResult(resJson);
            } catch (err) {
                alert('Error: ' + err.message);
                console.error(err);
            }

            return;
        }

        // Else regression-form: serialize to JSON
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((value, key) => payload[key] = value);

        try {
            const resp = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const resJson = await resp.json();
            if (!resp.ok) throw new Error(resJson.error || 'Prediction failed');
            showPredictionResult(resJson);
        } catch (err) {
            alert('Error: ' + err.message);
            console.error(err);
        }
    });
});