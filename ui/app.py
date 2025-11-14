"""
PERSON 1: AMIT - UI Application
Handles: Web interface, file upload, status display, visualization gallery
"""
import json
import os
import threading
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from config.config import (
    OUTPUT_DIR, RAW_DIR, PLOTS_DIR, STATUS_FILE,
    SCHEMA_FILE, PLOT_METADATA_FILE, INSIGHTS_FILE,
    DATA_QUALITY_REPORT_FILE, ANALYSIS_REPORT_FILE,
    TRANSFORMATION_LOG_FILE, CLEANED_DIR, REPORTS_DIR, LOGS_DIR
)
from core.ceo import ceo, AgentRequest
from agents import DataAgent, PlotAgent, AnalysisAgent


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = RAW_DIR

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'pdf', 'txt'}


# Register agents with CEO on module load
def _setup_agents():
    """Register all agents with CEO when the app module is imported"""
    if not ceo.agents:  # Only register if not already done
        data_agent = DataAgent()
        plot_agent = PlotAgent()
        analysis_agent = AnalysisAgent()

        ceo.register_agent("data", data_agent)
        ceo.register_agent("plot", plot_agent)
        ceo.register_agent("analysis", analysis_agent)
        print("[UI] All agents registered with CEO")


_setup_agents()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_data():
    """
    Delete all previous pipeline outputs before starting new run.
    This ensures a clean slate for each new file upload.
    """
    print("[UI] Cleaning up old data from previous runs...")

    # Directories to clean (delete contents, recreate empty)
    directories_to_clean = [CLEANED_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR]

    # Individual files to delete
    files_to_delete = [
        STATUS_FILE,
        SCHEMA_FILE,
        PLOT_METADATA_FILE,
        INSIGHTS_FILE,
        DATA_QUALITY_REPORT_FILE,
        ANALYSIS_REPORT_FILE,
        TRANSFORMATION_LOG_FILE
    ]

    # Delete and recreate directories
    for directory in directories_to_clean:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                print(f"[UI] Deleted directory: {directory}")
            except Exception as e:
                print(f"[UI] Error deleting directory {directory}: {e}")

        # Recreate empty directory
        directory.mkdir(parents=True, exist_ok=True)

    # Delete individual files
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"[UI] Deleted file: {file_path}")
            except Exception as e:
                print(f"[UI] Error deleting file {file_path}: {e}")

    print("[UI] Cleanup completed.")


# ========================================
# AMIT: TODO - Implement these endpoints
# ========================================

@app.route('/')
def index():
    """
    Render main dashboard

    OUTPUT:
        - HTML page with file upload, status, and results display

    TODO: Create HTML template with:
    - File upload component (drag-drop)
    - Real-time status display
    - Data preview panel
    - Visualization gallery
    - Insights panel
    - Retry buttons
    """
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload

    INPUT:
        - POST request with file in form data

    OUTPUT:
        - JSON {"success": bool, "message": str, "file_path": str}

    TODO: Implement file upload handling
    - Validate file type
    - Save to RAW_DIR
    - Trigger CEO pipeline
    - Return file path
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "message": f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        }), 400

    # Save file
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = RAW_DIR / filename
    file.save(filepath)

    # Clean up old data from previous runs
    cleanup_old_data()

    # Trigger CEO pipeline in background thread
    def run_pipeline_async():
        try:
            print(f"[UI] Starting pipeline for {filepath}")
            result = ceo.run_pipeline(str(filepath))
            print(f"[UI] Pipeline completed: {result}")
        except Exception as e:
            print(f"[UI] Pipeline error: {e}")

    # Start pipeline in background thread
    pipeline_thread = threading.Thread(target=run_pipeline_async)
    pipeline_thread.daemon = True
    pipeline_thread.start()

    print(f"[UI] File uploaded and pipeline started for {filepath}")

    return jsonify({
        "success": True,
        "message": f"File uploaded successfully: {filename}. Pipeline started.",
        "file_path": str(filepath)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get current pipeline status

    OUTPUT:
        - JSON with status from status.json

    TODO: Read status.json and return current state
    - Poll this endpoint from frontend for real-time updates
    """
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            status = json.load(f)
        return jsonify(status)
    else:
        return jsonify({
            "current_stage": None,
            "stages": {},
            "errors": [],
            "retry_count": 0
        })


@app.route('/api/schema', methods=['GET'])
def get_schema():
    """
    Get inferred schema

    OUTPUT:
        - JSON with schema from schema.json
    """
    if SCHEMA_FILE.exists():
        with open(SCHEMA_FILE, 'r') as f:
            schema = json.load(f)
        return jsonify(schema)
    else:
        return jsonify({"error": "Schema not available"}), 404


@app.route('/api/plots', methods=['GET'])
def get_plots():
    """
    Get plot metadata

    OUTPUT:
        - JSON with plot metadata from plot_metadata.json
    """
    if PLOT_METADATA_FILE.exists():
        with open(PLOT_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    else:
        return jsonify({"error": "Plot metadata not available"}), 404


@app.route('/api/insights', methods=['GET'])
def get_insights():
    """
    Get analysis insights

    OUTPUT:
        - JSON with insights from insights.json
    """
    if INSIGHTS_FILE.exists():
        with open(INSIGHTS_FILE, 'r') as f:
            insights = json.load(f)
        return jsonify(insights)
    else:
        return jsonify({"error": "Insights not available"}), 404


@app.route('/api/reports/quality', methods=['GET'])
def get_quality_report():
    """
    Get data quality report

    OUTPUT:
        - Markdown text
    """
    if DATA_QUALITY_REPORT_FILE.exists():
        with open(DATA_QUALITY_REPORT_FILE, 'r') as f:
            report = f.read()
        return report, 200, {'Content-Type': 'text/markdown'}
    else:
        return "Report not available", 404


@app.route('/api/reports/analysis', methods=['GET'])
def get_analysis_report():
    """
    Get analysis report

    OUTPUT:
        - Markdown text
    """
    if ANALYSIS_REPORT_FILE.exists():
        with open(ANALYSIS_REPORT_FILE, 'r') as f:
            report = f.read()
        return report, 200, {'Content-Type': 'text/markdown'}
    else:
        return "Report not available", 404


@app.route('/api/transformation_log', methods=['GET'])
def get_transformation_log():
    """
    Get transformation log

    OUTPUT:
        - JSON with transformation log
    """
    if TRANSFORMATION_LOG_FILE.exists():
        with open(TRANSFORMATION_LOG_FILE, 'r') as f:
            log = json.load(f)
        return jsonify(log)
    else:
        return jsonify({"error": "Transformation log not available"}), 404


@app.route('/plots/<filename>')
def serve_plot(filename):
    """
    Serve plot images

    INPUT:
        - filename: Plot filename

    OUTPUT:
        - Image file
    """
    return send_from_directory(PLOTS_DIR, filename)


@app.route('/api/retry/<stage>', methods=['POST'])
def retry_stage(stage):
    """
    Retry a failed stage

    INPUT:
        - stage: Stage name (data, plot, analysis)

    OUTPUT:
        - JSON {"success": bool, "message": str}
    """
    print(f"[UI] Retrying stage: {stage}")

    # Validate stage name
    valid_stages = ['data', 'plot', 'analysis']
    if stage not in valid_stages:
        return jsonify({
            "success": False,
            "message": f"Invalid stage: {stage}. Valid stages: {valid_stages}"
        }), 400

    # Get the last processed file from status
    if not STATUS_FILE.exists():
        return jsonify({
            "success": False,
            "message": "No active pipeline to retry"
        }), 400

    try:
        with open(STATUS_FILE, 'r') as f:
            status_data = json.load(f)

        # Find the file path from the most recent run
        # For simplicity, we'll look for the most recent file in RAW_DIR
        raw_files = list(RAW_DIR.glob('*'))
        if not raw_files:
            return jsonify({
                "success": False,
                "message": "No uploaded files found"
            }), 400

        latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)

        # Retry the stage in background thread
        def retry_async():
            try:
                print(f"[UI] Retrying {stage} stage for {latest_file}")

                if stage == 'data':
                    request = AgentRequest("data", "process_file", {"file_path": str(latest_file)})
                    response = ceo.execute_with_retry(request)
                elif stage == 'plot':
                    # Need data from previous stage
                    if SCHEMA_FILE.exists():
                        request = AgentRequest("plot", "create_plots", {"file_path": str(latest_file)})
                        response = ceo.execute_with_retry(request)
                    else:
                        print(f"[UI] Cannot retry plot: data stage not completed")
                        return
                elif stage == 'analysis':
                    request = AgentRequest("analysis", "analyze", {"file_path": str(latest_file)})
                    response = ceo.execute_with_retry(request)

                print(f"[UI] Retry completed for {stage}: {response.success}")

            except Exception as e:
                print(f"[UI] Retry error for {stage}: {e}")

        retry_thread = threading.Thread(target=retry_async)
        retry_thread.daemon = True
        retry_thread.start()

        return jsonify({
            "success": True,
            "message": f"Retry initiated for {stage} stage"
        })

    except Exception as e:
        print(f"[UI] Error in retry endpoint: {e}")
        return jsonify({
            "success": False,
            "message": f"Retry failed: {str(e)}"
        }), 500


# TODO: Create HTML template at ui/templates/index.html
# This should include:
# 1. File upload area (drag-drop)
# 2. Status display (polls /api/status)
# 3. Schema viewer (displays /api/schema)
# 4. Quality report viewer (displays /api/reports/quality)
# 5. Plot gallery (loads plots from /api/plots)
# 6. Insights panel (displays /api/insights)
# 7. Retry buttons (calls /api/retry/<stage>)


if __name__ == '__main__':
    # Create output directories
    for dir_path in [RAW_DIR, PLOTS_DIR.parent]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("""
    ====================================
    Multi-Agent Data Command Center UI
    ====================================

    TODO for AMIT:
    1. Create ui/templates/index.html
    2. Add JavaScript for:
       - File upload (drag-drop)
       - Status polling (every 2 seconds)
       - Data display
       - Plot gallery
       - Insights panel
    3. Implement retry buttons
    4. Add loading indicators
    5. Style with CSS/TailwindCSS

    ====================================
    """)

    app.run(debug=True, port=5000)
