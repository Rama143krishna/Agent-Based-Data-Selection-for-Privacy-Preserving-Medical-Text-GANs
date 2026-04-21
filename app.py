# ===============================================================
# AGENTIC_AI_PROJECT - Main Flask Application
# ===============================================================

from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import os
import io
import zipfile
from datetime import datetime
import traceback

app = Flask(__name__)

# ===============================================================
# CONFIGURATION
# ===============================================================

OUTPUT_FOLDERS = {
    'RF': 'RF',
    'HYBRID': 'HYBRID',
    'HYDRA': 'HYDRA'
}

for folder in OUTPUT_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)


# ===============================================================
# HOME ROUTE
# ===============================================================

@app.route('/')
def index():
    return render_template('index.html')


# ===============================================================
# BLOCK 1: RANDOM FOREST - READ ENTIRE CSV FILES
# ===============================================================

@app.route('/api/run_block1', methods=['POST'])
def run_block1():
    """Execute Random Forest Privacy Agent"""
    try:
        print("\n" + "=" * 100)
        print("🌲 BLOCK 1: RANDOM FOREST PRIVACY AGENT - RUNNING")
        print("=" * 100)

        import random_forest
        result = random_forest.run_random_forest()

        if result['status'] == 'success':
            raw_data_path = "RF/RAW_DATA.csv"
            privacy_results_path = "RF/RF_PRIVACY_RESULTS.csv"
            safe_text_path = "RF/SAFE_TEXT_FOR_LLMS.csv"

            raw_df = pd.read_csv(raw_data_path)
            privacy_df = pd.read_csv(privacy_results_path)
            safe_df = pd.read_csv(safe_text_path)

            raw_data_display = raw_df.to_html(classes="data-table", index=False)
            privacy_display = privacy_df.to_html(classes="data-table", index=False)
            safe_text_display = safe_df.to_html(classes="data-table", index=False)

            result['raw_data'] = raw_data_display
            result['privacy_results'] = privacy_display
            result['safe_text'] = safe_text_display

            result['raw_data_rows'] = len(raw_df)
            result['privacy_results_rows'] = len(privacy_df)
            result['safe_text_rows'] = len(safe_df)

        return jsonify(result)

    except Exception as e:
        print(f"❌ ERROR in Block 1: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Block 1 failed - Check dataset path'
        }), 500


# ===============================================================
# BLOCK 2: HYBRID MODEL - READ ENTIRE CSV FILES
# ===============================================================

@app.route('/api/run_block2', methods=['POST'])
def run_block2():
    """Execute Hybrid BioGPT + LLAMA Model"""
    try:
        print("\n" + "=" * 100)
        print("🧠 BLOCK 2: HYBRID BIOGPT + LLAMA - RUNNING")
        print("=" * 100)

        import hybrid_model
        result = hybrid_model.run_hybrid_model()

        if result['status'] == 'success':
            rf_vs_llm_path = "HYBRID/RF_VS_LLM_DIFF.csv"
            if os.path.exists(rf_vs_llm_path):
                rf_vs_llm_df = pd.read_csv(rf_vs_llm_path)
                rf_vs_llm_display = rf_vs_llm_df.to_html(classes="data-table", index=False)
                result['rf_vs_llm_display'] = rf_vs_llm_display
                result['rf_vs_llm_rows'] = len(rf_vs_llm_df)

            examples_path = "HYBRID/20_SAMPLE_EXAMPLES.csv"
            if os.path.exists(examples_path):
                examples_df = pd.read_csv(examples_path)
                examples_display = examples_df.to_html(classes="data-table", index=False)
                result['examples_display'] = examples_display
                result['examples_rows'] = len(examples_df)

            approved_path = "HYBRID/APPROVED_TEXT_FOR_GAN.csv"
            if os.path.exists(approved_path):
                approved_df = pd.read_csv(approved_path)
                approved_display = approved_df.to_html(classes="data-table", index=False)
                result['approved_text_display'] = approved_display
                result['approved_rows'] = len(approved_df)

        return jsonify(result)

    except Exception as e:
        print(f"❌ ERROR in Block 2: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Block 2 failed - Make sure Block 1 completed first'
        }), 500


# ===============================================================
# BLOCK 3: HYDRAGAN - READ ENTIRE CSV FILES
# ===============================================================

@app.route('/api/run_block3', methods=['POST'])
def run_block3():
    """Execute HydraGAN Synthetic Data Generation"""
    try:
        print("\n" + "="*100)
        print("🎨 BLOCK 3: HYDRAGAN SYNTHETIC DATA GENERATION - RUNNING")
        print("="*100)

        import hydragan
        result = hydragan.run_hydragan()

        if result['status'] == 'success':
            # ✅ SYNTHETIC TEXT - CONVERT TO HTML TABLE
            synthetic_path = "HYDRA/HYDRAGAN_SYNTHETIC_TEXT_FINAL.csv"
            if os.path.exists(synthetic_path):
                synthetic_df = pd.read_csv(synthetic_path)
                # ✅ CONVERT DATAFRAME TO HTML TABLE
                synthetic_display = synthetic_df.to_html(classes="data-table", index=False)
                result['synthetic_display'] = synthetic_display
                result['synthetic_rows'] = len(synthetic_df)

            # ✅ PERFORMANCE METRICS - CONVERT TO DICTIONARY
            metrics_path = "HYDRA/HYDRAGAN_METRICS_FINAL.csv"
            if os.path.exists(metrics_path):
                metrics_df = pd.read_csv(metrics_path)
                # Convert DataFrame to dictionary
                metrics_dict = {}
                for idx, row in metrics_df.iterrows():
                    metric_name = row['Metric']
                    metric_value = float(row['Value'])
                    metrics_dict[metric_name] = metric_value
                result['metrics'] = metrics_dict

           # Print to terminal
            print("\n" + "="*100)
            print("🤖 SYNTHETIC TEXT OUTPUT (First 10 samples):")
            print("="*100)
            if os.path.exists(synthetic_path):
                print(pd.read_csv(synthetic_path).head(10).to_string())

            print("\n" + "="*100)
            print("🏆 HYDRAGAN PERFORMANCE METRICS:")
            print("="*100)
            if os.path.exists(metrics_path):
                print(pd.read_csv(metrics_path).to_string())

            print("\n" + "="*100)
            print("✓ BLOCK 3 COMPLETED SUCCESSFULLY")
            print("="*100 + "\n")

        return jsonify(result)

    except Exception as e:
        print(f"❌ ERROR in Block 3: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Block 3 failed - Make sure Block 2 completed first'
        }), 500

# ===============================================================
# DOWNLOAD ENDPOINTS
# ===============================================================

@app.route('/api/download_block1', methods=['GET'])
def download_block1():
    """Download Block 1 outputs as ZIP"""
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            rf_folder = OUTPUT_FOLDERS['RF']
            if os.path.exists(rf_folder):
                for file in os.listdir(rf_folder):
                    file_path = os.path.join(rf_folder, file)
                    if os.path.isfile(file_path):
                        zf.write(file_path, arcname=file)
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'block1_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_block2', methods=['GET'])
def download_block2():
    """Download Block 2 outputs as ZIP"""
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            hybrid_folder = OUTPUT_FOLDERS['HYBRID']
            if os.path.exists(hybrid_folder):
                for file in os.listdir(hybrid_folder):
                    file_path = os.path.join(hybrid_folder, file)
                    if os.path.isfile(file_path):
                        zf.write(file_path, arcname=file)
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'block2_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_block3', methods=['GET'])
def download_block3():
    """Download Block 3 outputs as ZIP"""
    try:
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            hydra_folder = OUTPUT_FOLDERS['HYDRA']
            if os.path.exists(hydra_folder):
                for file in os.listdir(hydra_folder):
                    file_path = os.path.join(hydra_folder, file)
                    if os.path.isfile(file_path):
                        zf.write(file_path, arcname=file)
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'block3_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===============================================================
# HEALTH CHECK
# ===============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# ===============================================================
# ERROR HANDLERS
# ===============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


# ===============================================================
# RUN APPLICATION
# ===============================================================

if __name__ == '__main__':
    print("\n" + "=" * 100)
    print("🚀 AGENTIC AI PROJECT - PRIVACY ML PIPELINE DASHBOARD")
    print("=" * 100)
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 API Base: http://localhost:5000/api")
    print("=" * 100 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)