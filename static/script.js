// ============================================
// BLOCK 1: RUN RANDOM FOREST
// ============================================

async function runBlock1() {
    const btn = event.target.closest('.run-btn');
    btn.classList.add('loading');

    updateStatusBar('status1', 'processing', '⏳ Running Random Forest...');

    try {
        console.log('🌲 Starting Block 1 - Random Forest...');

        const response = await fetch('/api/run_block1', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        console.log('✓ Block 1 Response:', data);

        // Display Raw Data HTML Table
        console.log('📊 Displaying Raw Data...');
        document.getElementById('block1-rawdata').innerHTML =
            `<div style="overflow-x: auto;">${data.raw_data || '<p style="color: red;">No data</p>'}</div>`;

        // Display Privacy Results HTML Table
        console.log('🔐 Displaying Privacy Results...');
        document.getElementById('block1-privacy').innerHTML =
            `<div style="overflow-x: auto;">${data.privacy_results || '<p style="color: red;">No data</p>'}</div>`;

        // Display Safe Text HTML Table
        console.log('✅ Displaying Safe Text...');
        document.getElementById('block1-safetext').innerHTML =
            `<div style="overflow-x: auto;">${data.safe_text || '<p style="color: red;">No data</p>'}</div>`;

        // Update row counts
        if (data.raw_data_rows) {
            document.getElementById('block1-raw-count').textContent = `(${data.raw_data_rows} rows)`;
        }
        if (data.privacy_results_rows) {
            document.getElementById('block1-privacy-count').textContent = `(${data.privacy_results_rows} rows)`;
        }
        if (data.safe_text_rows) {
            document.getElementById('block1-safe-count').textContent = `(${data.safe_text_rows} rows)`;
        }

        // Display Metrics Table
        console.log('📈 Displaying Metrics...');
        displayMetricsTable('block1-metrics', data.metrics);

        updateStatusBar('status1', 'success', '✓ Block 1 Completed');
        console.log('✓ Block 1 Complete');

    } catch (error) {
        console.error('❌ Block 1 Error:', error);
        document.getElementById('block1-rawdata').innerHTML =
            `<p style="color: red; font-weight: bold;">❌ ERROR: ${error.message}</p>`;
        updateStatusBar('status1', 'error', '✗ Block 1 Failed');
    }

    btn.classList.remove('loading');
}

function downloadBlock1() {
    console.log('⬇ Downloading Block 1...');
    fetch('/api/download_block1')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'block1_output.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            console.log('✓ Download complete');
        })
        .catch(error => {
            console.error('❌ Download failed:', error);
            alert('❌ Download failed: ' + error.message);
        });
}

// ============================================
// BLOCK 2: RUN HYBRID MODEL
// ============================================

async function runBlock2() {
    const btn = event.target.closest('.run-btn');
    btn.classList.add('loading');

    updateStatusBar('status2', 'processing', '⏳ Running Hybrid Model...');

    try {
        console.log('🧠 Starting Block 2 - Hybrid Model...');

        const response = await fetch('/api/run_block2', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        console.log('✓ Block 2 Response:', data);

        // Display RF vs LLM Comparison
        console.log('⚖️ Displaying RF vs LLM...');
        if (data.rf_vs_llm_display) {
            document.getElementById('block2-rf-vs-llm').innerHTML =
                `<div style="overflow-x: auto;">${data.rf_vs_llm_display}</div>`;
            if (data.rf_vs_llm_rows) {
                document.getElementById('block2-rf-vs-llm-count').textContent = `(${data.rf_vs_llm_rows} rows)`;
            }
        }

        // Display Overall Performance
        console.log('🎯 Displaying Overall Performance...');
        if (data.overall_performance) {
            displayMetricsTable('block2-overall-performance', data.overall_performance);
        }

        // Display Model Scores
        console.log('🧠 Displaying Model Scores...');
        if (data.model_scores) {
            displayModelScoresTable('block2-model-scores', data.model_scores);
        }

        // Display 20 Examples
        console.log('📋 Displaying 20 Examples...');
        if (data.examples_display) {
            displayExamplesWithFormat('block2-examples', data.examples_display);
            if (data.examples_rows) {
                document.getElementById('block2-examples-count').textContent = `(20 examples)`;
            }
        }

        // Display Approved Text
        console.log('✔️ Displaying Approved Text...');
        if (data.approved_text_display) {
            document.getElementById('block2-approved').innerHTML =
                `<div style="overflow-x: auto;">${data.approved_text_display}</div>`;
            if (data.approved_rows) {
                document.getElementById('block2-approved-count').textContent = `(${data.approved_rows} rows)`;
            }
        }

        updateStatusBar('status2', 'success', '✓ Block 2 Completed');
        console.log('✓ Block 2 Complete');

    } catch (error) {
        console.error('❌ Block 2 Error:', error);
        document.getElementById('block2-rf-vs-llm').innerHTML =
            `<p style="color: red; font-weight: bold;">❌ ERROR: ${error.message}</p>`;
        updateStatusBar('status2', 'error', '✗ Block 2 Failed');
    }

    btn.classList.remove('loading');
}

function downloadBlock2() {
    console.log('⬇ Downloading Block 2...');
    fetch('/api/download_block2')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'block2_output.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            console.log('✓ Download complete');
        })
        .catch(error => {
            console.error('❌ Download failed:', error);
            alert('❌ Download failed: ' + error.message);
        });
}

// ============================================
// BLOCK 3: RUN HYDRAGAN
// ============================================

async function runBlock3() {
    const btn = event.target.closest('.run-btn');
    btn.classList.add('loading');

    updateStatusBar('status3', 'processing', '⏳ Running HydraGAN...');

    try {
        console.log('🎨 Starting Block 3 - HydraGAN...');

        const response = await fetch('/api/run_block3', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        console.log('✓ Block 3 Response:', data);

        // Display Synthetic Text
        console.log('🤖 Displaying Synthetic Text...');
        if (data.synthetic_display) {
            document.getElementById('block3-synthetic').innerHTML =
                `<div style="overflow-x: auto;">${data.synthetic_display}</div>`;
            if (data.synthetic_rows) {
                document.getElementById('block3-synthetic-count').textContent = `(${data.synthetic_rows} rows)`;
            }
        }

        // Display HydraGAN Performance Metrics with Colors
        console.log('🏆 Displaying Performance Metrics...');
        if (data.metrics) {
            displayHydraganMetrics('block3-performance', data.metrics);
        }

        updateStatusBar('status3', 'success', '✓ Block 3 Completed');
        console.log('✓ Block 3 Complete');

    } catch (error) {
        console.error('❌ Block 3 Error:', error);
        document.getElementById('block3-synthetic').innerHTML =
            `<p style="color: red; font-weight: bold;">❌ ERROR: ${error.message}</p>`;
        updateStatusBar('status3', 'error', '✗ Block 3 Failed');
    }

    btn.classList.remove('loading');
}

function downloadBlock3() {
    console.log('⬇ Downloading Block 3...');
    fetch('/api/download_block3')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'block3_output.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            console.log('✓ Download complete');
        })
        .catch(error => {
            console.error('❌ Download failed:', error);
            alert('❌ Download failed: ' + error.message);
        });
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function updateStatusBar(elementId, status, message) {
    const statusBar = document.getElementById(elementId);
    statusBar.className = `status-bar ${status}`;
    statusBar.innerHTML = `<span class="status-dot"></span> ${message}`;
}

function displayMetricsTable(tableId, metricsData) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = '';

    if (!metricsData) return;

    Object.entries(metricsData).forEach(([key, value]) => {
        const row = tbody.insertRow();
        const cell1 = row.insertCell(0);
        const cell2 = row.insertCell(1);

        cell1.textContent = key;
        cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
    });
}

function displayModelScoresTable(tableId, modelScoresData) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = '';

    if (!modelScoresData) return;

    const colors = ['#ec4899', '#3b82f6', '#10b981'];
    let colorIndex = 0;

    Object.entries(modelScoresData).forEach(([key, value]) => {
        const row = tbody.insertRow();
        row.style.borderLeftColor = colors[colorIndex];
        row.style.borderLeftWidth = '4px';
        row.style.borderLeftStyle = 'solid';

        const cell1 = row.insertCell(0);
        const cell2 = row.insertCell(1);

        cell1.textContent = key;
        cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
        cell2.style.color = colors[colorIndex];
        cell2.style.fontWeight = '600';

        colorIndex++;
    });
}

function displayExamplesWithFormat(elementId, htmlContent) {
    const container = document.getElementById(elementId);

    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');
    const rows = doc.querySelectorAll('tbody tr');

    let examplesHtml = '';

    rows.forEach((row, index) => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 2) {
            const rfOutput = cells[0].textContent;
            const llmOutput = cells[1].textContent;

            examplesHtml += `
                <div class="example-item">
                    <div class="example-header">Example ${index + 1}:</div>
                    <div class="example-before">
                        <div class="example-before-label">RF Output:</div>
                        <div class="example-before-text">${rfOutput}</div>
                    </div>
                    <div>
                        <div class="example-after-label">LLM Output:</div>
                        <div class="example-after-text">${llmOutput}</div>
                    </div>
                    <div class="example-divider">─────────────────────────────────────────────</div>
                </div>
            `;
        }
    });

    container.innerHTML = examplesHtml || '<p style="color: red;">No examples available</p>';
}

// ✅ DISPLAY HYDRAGAN METRICS TABLE WITH COLORS
function displayHydraganMetrics(tableId, metricsData) {
    const element = document.getElementById(tableId);

    if (!element) {
        console.error(`Table element with ID ${tableId} not found!`);
        return;
    }

    const tbody = element.querySelector('tbody');
    tbody.innerHTML = '';

    if (!metricsData || Object.keys(metricsData).length === 0) {
        console.warn('No metrics data provided');
        return;
    }

    // Define colors for each metric in order
    const colors = [
        '#6366f1',  // BLEU Score - Purple
        '#ec4899',  // ROUGE Score - Pink
        '#f59e0b',  // Distinct-1 - Orange
        '#3b82f6',  // Distinct-2 - Blue
        '#10b981',  // Self-BLEU - Green
        '#10b981'   // Privacy Leakage - Green (Safe)
    ];

    // Define metric order
    const metricOrder = [
        'BLEU Score',
        'ROUGE Score',
        'Distinct-1',
        'Distinct-2',
        'Self-BLEU',
        'Privacy Leakage Rate'
    ];

    let colorIndex = 0;

    // Display metrics in specific order
    metricOrder.forEach(metricName => {
        if (metricName in metricsData) {
            const row = tbody.insertRow();
            row.style.borderLeftColor = colors[colorIndex];
            row.style.borderLeftWidth = '4px';
            row.style.borderLeftStyle = 'solid';
            row.style.transition = 'all 0.3s ease';

            const cell1 = row.insertCell(0);
            const cell2 = row.insertCell(1);

            cell1.textContent = metricName;
            cell1.style.fontWeight = '600';
            cell1.style.color = 'var(--text-primary)';

            const value = metricsData[metricName];
            cell2.textContent = typeof value === 'number' ? value.toFixed(4) : value;
            cell2.style.color = colors[colorIndex];
            cell2.style.fontWeight = '700';
            cell2.style.fontSize = '1.05em';

            colorIndex++;
        }
    });
}

console.log('✅ Script loaded successfully');