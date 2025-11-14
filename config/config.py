"""
Configuration for Multi-Agent Data Command Center
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Default model (fastest & cheapest for testing)

# Available Claude Models
AVAILABLE_MODELS = {
    "sonnet-4.5": "claude-sonnet-4-5-20250929",
    "sonnet-4": "claude-sonnet-4-20250514",
    "haiku-4.5": "claude-haiku-4-5-20251001",
    "opus-4.1": "claude-opus-4-1-20250805",
    "opus-4": "claude-opus-4-20250514"
}

# Per-Agent Model Configuration
# Configure which model each agent should use (defaults to Haiku for testing)
AGENT_MODELS = {
    "data": os.getenv("DATA_AGENT_MODEL", "claude-haiku-4-5-20251001"),
    "plot": os.getenv("PLOT_AGENT_MODEL", "claude-haiku-4-5-20251001"),
    "analysis": os.getenv("ANALYSIS_AGENT_MODEL", "claude-haiku-4-5-20251001")
}

# Extended Thinking Configuration
# Enable extended thinking (reasoning) for more complex tasks
EXTENDED_THINKING = {
    "enabled": os.getenv("EXTENDED_THINKING_ENABLED", "false").lower() == "true",
    "budget_tokens": int(os.getenv("THINKING_BUDGET_TOKENS", "5000"))  # Min: 1024
}

# Agent-specific extended thinking overrides
AGENT_THINKING_CONFIG = {
    "data": {
        "enabled": os.getenv("DATA_AGENT_THINKING", "false").lower() == "true",
        "budget_tokens": int(os.getenv("DATA_AGENT_THINKING_BUDGET", "5000"))
    },
    "plot": {
        "enabled": os.getenv("PLOT_AGENT_THINKING", "false").lower() == "true",
        "budget_tokens": int(os.getenv("PLOT_AGENT_THINKING_BUDGET", "5000"))
    },
    "analysis": {
        "enabled": os.getenv("ANALYSIS_AGENT_THINKING", "true").lower() == "true",  # Default ON for analysis
        "budget_tokens": int(os.getenv("ANALYSIS_AGENT_THINKING_BUDGET", "8000"))  # Higher budget for complex analysis
    }
}

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
