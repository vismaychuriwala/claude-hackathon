"""
Configuration for Multi-Agent Data Command Center
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# Claude API configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")  # Set via environment variable
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"  # Latest Sonnet model

# Agent configuration
MAX_RETRIES = 3
TIMEOUT_SECONDS = 300

# File paths for agent communication
STATUS_FILE = LOGS_DIR / "status.json"
SCHEMA_FILE = OUTPUT_DIR / "schema.json"
TRANSFORMATION_LOG_FILE = OUTPUT_DIR / "transformation_log.json"
PLOT_METADATA_FILE = OUTPUT_DIR / "plot_metadata.json"
INSIGHTS_FILE = OUTPUT_DIR / "insights.json"
DATA_QUALITY_REPORT_FILE = REPORTS_DIR / "data_quality_report.md"
ANALYSIS_REPORT_FILE = REPORTS_DIR / "analysis_report.md"
