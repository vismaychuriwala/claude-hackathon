# Multi-Agent Data Command Center - How It Works

## Quick Start: How to Run

### 1. Setup (One-time)

```bash
# Navigate to project
cd /home/vismay/claude-hackathon

# Activate virtual environment and install dependencies
source activate.sh

# Add your Claude API key
echo "CLAUDE_API_KEY=your-api-key-here" > .env
```

### 2. Run the Pipeline

```bash
# Option 1: Run on test data
python main.py test_data.csv

# Option 2: Run on your own CSV file
python main.py path/to/your/data.csv

# Option 3: Start the web UI (Amit's work)
python main.py ui
```

### 3. View the Outputs

After running, check these files:

```bash
# View the schema Claude inferred
cat output/schema.json

# View the data quality report
cat output/reports/data_quality_report.md

# View what transformations were applied
cat output/transformation_log.json

# View the cleaned data
cat output/cleaned/cleaned_data.csv

# View generated plots
ls output/plots/

# View analysis insights
cat output/insights.json
```

### 4. Expected Output

You should see:

```
[CEO] Starting pipeline for: test_data.csv
[CEO] Stage 1/3: Data Processing
[DataAgent] Inferring schema with Claude...
[DataAgent] ✓ Schema inferred: 5 columns, 6 warnings
[DataAgent] Generating cleaning strategy with Claude...
[DataAgent] ✓ Cleaning complete: 2 operations applied
[DataAgent] Generating report with Claude...
[DataAgent] ✓ Report generated
[CEO] ✓ Data processing complete

[CEO] Stage 2/3: Plot Generation
[PlotAgent] Creating plots...
[CEO] ✓ Plot generation complete

[CEO] Stage 3/3: Analysis
[AnalysisAgent] Generating insights...
[CEO] ✓ Analysis complete

[CEO] ✓ Pipeline completed successfully!
```

---

## Architecture Overview

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                           │
│                    (uploads CSV file)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    CEO ORCHESTRATOR                          │
│              (Central Coordinator)                           │
│   - Routes all requests                                      │
│   - Handles retries (max 3 attempts)                        │
│   - Logs status to status.json                             │
│   - Manages workflow: Data → Plot → Analysis                │
└─────┬────────────────┬────────────────┬─────────────────────┘
      │                │                │
      ▼                ▼                ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│   DATA   │    │   PLOT   │    │ ANALYSIS │
│  AGENT   │───▶│  AGENT   │───▶│  AGENT   │
│ (Vismay) │    │ (Nikunj) │    │(Shamanth)│
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT FILES                              │
│  • schema.json                                               │
│  • cleaned_data.csv                                          │
│  • transformation_log.json                                   │
│  • data_quality_report.md                                    │
│  • plots/*.png                                               │
│  • plot_metadata.json                                        │
│  • insights.json                                             │
│  • analysis_report.md                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      WEB UI (Flask)                          │
│                      (Amit's work)                           │
│   - Displays all outputs                                     │
│   - Real-time status updates                                 │
│   - Interactive visualizations                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. CEO Orchestrator (`core/ceo.py`)

**Purpose:** Central coordinator that routes all agent actions and manages the workflow.

**Key Classes:**

#### `AgentRequest`
Represents a request to an agent:
```python
AgentRequest(
    agent_name="data",           # Which agent to call
    action="process_file",       # What action to perform
    data={"file_path": "..."}    # Input data
)
```

#### `AgentResponse`
Represents a response from an agent:
```python
AgentResponse(
    success=True,                # Did it succeed?
    data={...},                  # Output data
    error=None,                  # Error message if failed
    retry_requested=False        # Should we retry?
)
```

#### `CEOOrchestrator`
The main coordinator:

**Methods:**
- `register_agent(name, agent_instance)` - Registers agents at startup
- `route_request(request)` - Routes a request to the appropriate agent
- `execute_with_retry(request)` - Executes with automatic retry (up to 3 times)
- `run_pipeline(file_path)` - **Main method** - Runs full 3-stage pipeline

**Pipeline Flow in `run_pipeline()`:**
```python
1. Stage 1: Data Processing
   - Creates AgentRequest for "data" agent
   - Executes with retry
   - If fails → abort pipeline
   - If succeeds → proceed to plotting

2. Stage 2: Plot Generation
   - Creates AgentRequest for "plot" agent
   - Passes cleaned data path and schema from Stage 1
   - If fails → log error but continue (non-critical)

3. Stage 3: Analysis
   - Creates AgentRequest for "analysis" agent
   - Passes all outputs from Stages 1 & 2
   - Generates final insights and reports

4. Return aggregated results
```

**Status Tracking:**
- Every operation updates `output/logs/status.json`
- UI polls this file for real-time progress
- Format:
  ```json
  {
    "current_stage": "plotting",
    "stages": {
      "data": {"status": "completed", "message": "..."},
      "plot": {"status": "in_progress", "message": "..."}
    }
  }
  ```

---

### 2. Data Agent (`agents/data_agent.py`) - **YOUR WORK**

**Purpose:** Handles data ingestion, schema inference, cleaning, and quality reporting.

**Main Method:** `process_file(file_path)`

This runs the full data pipeline:
```
Input CSV → Ingest → Preprocess → Infer Schema → Clean Data → Generate Report → Save Outputs
```

#### Step-by-Step Breakdown:

##### **Step 1: `_ingest_file(file_path)`**
- **Current:** Detects file type (placeholder: assumes CSV)
- **Returns:** `{"type": "csv", "size_bytes": 1234, "encoding": "utf-8"}`
- **TODO (Low Priority):** Add support for Excel, JSON, PDF

##### **Step 2: `_preprocess(file_path, file_info)`**
- **Current:** Reads CSV into pandas DataFrame
- **Returns:** DataFrame
- **TODO (Low Priority):** Handle Excel, PDF table extraction

##### **Step 3: `_infer_schema(df)` - ⭐ CLAUDE-POWERED**
This is where the magic happens!

**What it does:**
1. Computes basic statistics for each column:
   - Null count and percentage
   - Unique value count
   - For numeric: min, max, mean, std
   - Sample values (first 10)

2. Sends this data to Claude with a prompt:
   ```
   "Analyze this dataset and provide intelligent type inference"
   ```

3. Claude responds with:
   - **Semantic type**: Not just "object" or "int64", but "categorical", "datetime", "id", "numeric", etc.
   - **Semantic meaning**: "Annual salary in USD", "Employee hire date", etc.
   - **Data quality**: "good", "warning", or "poor"
   - **Specific issues**: "Should be converted to datetime type", "High null percentage", etc.
   - **Overall warnings**: Dataset-level issues

4. Merges Claude's analysis with statistical data

**Example Output:**
```json
{
  "columns": [
    {
      "name": "join_date",
      "type": "object",
      "inferred_type": "datetime",
      "semantic_meaning": "Employee hire date in YYYY-MM-DD format",
      "null_pct": 0.0,
      "unique_count": 5,
      "data_quality": "warning",
      "issues": [
        "Stored as string instead of datetime type",
        "Should be converted to datetime64"
      ]
    }
  ],
  "warnings": [
    "Very small dataset (5 rows) - statistical measures may not be reliable",
    "Column 'join_date' should be converted to datetime type"
  ]
}
```

**Fallback:** If Claude API fails, returns basic schema with pandas dtypes.

##### **Step 4: `_clean_data(df, schema)` - ⭐ CLAUDE-POWERED**

**What it does:**
1. Prepares schema summary for Claude
2. Asks Claude: "Based on this schema analysis, recommend a cleaning strategy"
3. Claude returns a list of operations:
   ```json
   {
     "operations": [
       {
         "operation": "normalize_categorical",
         "column": "department",
         "reason": "Ensure consistency to prevent duplicate categories",
         "parameters": {}
       },
       {
         "operation": "impute_missing",
         "column": "salary",
         "reason": "Fill missing values with median",
         "parameters": {"method": "median"}
       }
     ]
   }
   ```

4. Executes each operation:
   - **drop_column**: Removes columns with >50% nulls
   - **impute_missing**: Fills nulls (median for numeric, mode for categorical)
   - **cast_type**: Converts strings to datetime, etc.
   - **remove_duplicates**: Removes exact duplicate rows
   - **handle_outliers**: Caps values beyond 3 standard deviations
   - **normalize_categorical**: Lowercase and trim whitespace

5. Logs every transformation:
   ```json
   {
     "operation": "normalize_categorical",
     "column": "department",
     "reason": "...",
     "rows_affected": 5
   }
   ```

**Fallback:** If Claude fails, applies basic duplicate removal.

##### **Step 5: `_generate_report(df, schema, transform_log)` - ⭐ CLAUDE-POWERED**

**What it does:**
1. Prepares comprehensive summary:
   - Original vs final row count
   - Columns dropped
   - Operations applied
   - Data quality warnings

2. Sends to Claude: "Generate a professional data quality report"

3. Claude generates a markdown report with:
   - **Executive Summary**: High-level overview with quality score
   - **Dataset Overview**: Dimensions, processing stats
   - **Data Quality Assessment**: Column-by-column analysis
   - **Transformations Applied**: Detailed operation log
   - **Warnings & Issues**: Critical problems highlighted
   - **Recommendations**: Actionable next steps

**Example Output:**
```markdown
# Data Quality Report

**Report Generated:** 2024
**Dataset Status:** ⚠️ Requires Attention

## Executive Summary

This dataset contains **5 rows** and **5 columns**. While data
completeness is perfect (100%), several critical concerns require
attention. The extremely small size limits statistical reliability,
and type conversion issues exist for temporal data.

**Overall Quality Score: 7.5/10**

**Key Findings:**
- ✅ Perfect data completeness
- ⚠️ Critical: Small sample size (5 rows)
- ⚠️ Date column requires type conversion
- ✅ Categorical normalization successful

## Data Quality Assessment

### name (Text)
- **Quality:** Good
- **Missing:** 0.0%
- **Unique Values:** 5
- **Issues:** None

### join_date (Datetime)
- **Quality:** Warning
- **Missing:** 0.0%
- **Issues:**
  - Stored as string instead of datetime type
  - Needs conversion for temporal operations

## Recommendations

1. Convert `join_date` to datetime64 type
2. Verify that small sample size is intentional
3. Add employee_id column if missing in full dataset
```

**Fallback:** Generates basic structured report if Claude fails.

##### **Step 6: `_save_outputs()`**

Saves everything to files:
- `output/cleaned/cleaned_data.csv` - Cleaned dataset
- `output/schema.json` - Schema with Claude's analysis
- `output/transformation_log.json` - Log of all operations
- `output/reports/data_quality_report.md` - Quality report

---

### 3. Plot Agent (`agents/plot_agent.py`) - **NIKUNJ'S WORK**

**Purpose:** Automatically generates visualizations from cleaned data.

**Status:** Currently has placeholder implementations with TODOs.

**What needs to be implemented:**

#### `_plan_plots(df, schema)` - TODO: Use Claude
Should use Claude to intelligently decide which plots to create:
- Histograms for numeric columns
- Bar charts for categorical columns
- Time series for datetime columns
- Correlation heatmaps
- Missingness patterns

#### `_generate_plot(df, plan)` - TODO: Implement
Should create plots using matplotlib/plotly based on the plan.

**Current Output:**
- Creates 1 basic histogram as placeholder
- Saves to `output/plots/`
- Generates `plot_metadata.json`

**Expected Output Structure:**
```json
{
  "plots": [
    {
      "filename": "histogram_age.png",
      "html_filename": "histogram_age.html",
      "type": "histogram",
      "column": "age",
      "description": "Distribution of ages showing normal distribution",
      "insights": ["Mean age: 30", "Range: 25-35"]
    }
  ]
}
```

---

### 4. Analysis Agent (`agents/analysis_agent.py`) - **SHAMANTH'S WORK**

**Purpose:** Interprets plots and data to generate actionable insights.

**Status:** Currently has placeholder implementations with TODOs.

**What needs to be implemented:**

#### `_statistical_analysis(df, schema)` - TODO: Use Claude
Should compute advanced statistics and use Claude to interpret them:
- Detect skewness and outliers
- Identify correlations
- Find anomalies

#### `_interpret_plots(df, schema, plot_metadata)` - TODO: Use Claude Vision
Could use Claude's vision API to analyze plot images and describe patterns.

#### `_business_insights(df, schema, stats)` - TODO: Use Claude
Should generate actionable business insights:
- Trends and patterns
- Recommendations
- Key findings

**Current Output:**
- Empty `insights.json`
- Basic `analysis_report.md`

**Expected Output Structure:**
```json
{
  "insights": [
    {
      "type": "anomaly",
      "severity": "high",
      "description": "3 orders with amount > $800 (3 std above mean)",
      "affected_columns": ["amount"],
      "affected_rows": [45, 234, 789],
      "recommendation": "Review manually or cap at 99th percentile"
    }
  ],
  "summary": {
    "total_insights": 3,
    "high_severity": 1,
    "recommended_next_steps": [...]
  }
}
```

---

### 5. Web UI (`ui/app.py` and `ui/templates/index.html`) - **AMIT'S WORK**

**Purpose:** Web interface to upload files and view all outputs.

**Flask Endpoints:**

- `GET /` - Main dashboard page
- `POST /api/upload` - Upload CSV file, triggers pipeline
- `GET /api/status` - Get current pipeline status (polls status.json)
- `GET /api/schema` - Get inferred schema
- `GET /api/plots` - Get list of generated plots
- `GET /api/insights` - Get analysis insights
- `GET /api/reports/quality` - Get data quality report
- `GET /api/reports/analysis` - Get analysis report
- `POST /api/retry/<stage>` - Retry a failed stage
- `GET /plots/<filename>` - Serve plot images

**How to start:**
```bash
python main.py ui
# Visit http://localhost:5000
```

**Current Status:** Basic endpoints implemented, frontend TODOs remain.

---

## File Structure

```
claude-hackathon/
├── main.py                    # Entry point - registers agents, runs pipeline
├── activate.sh                # Quick activation script
├── requirements.txt           # Python dependencies
├── .env                       # Your API key (NOT committed to git)
├── .env.example              # Template (committed to git)
│
├── config/
│   └── config.py             # All paths and settings
│
├── core/
│   └── ceo.py                # CEO Orchestrator
│
├── agents/
│   ├── __init__.py
│   ├── data_agent.py         # Data Agent (YOUR WORK)
│   ├── plot_agent.py         # Plot Agent (NIKUNJ)
│   └── analysis_agent.py     # Analysis Agent (SHAMANTH)
│
├── ui/
│   ├── app.py                # Flask backend (AMIT)
│   └── templates/
│       └── index.html        # Frontend (AMIT)
│
├── utils/
│   └── claude_client.py      # Claude API wrapper
│
└── output/                   # Generated files
    ├── cleaned/
    │   └── cleaned_data.csv
    ├── plots/
    │   └── *.png
    ├── reports/
    │   ├── data_quality_report.md
    │   └── analysis_report.md
    ├── logs/
    │   └── status.json
    ├── schema.json
    ├── transformation_log.json
    ├── plot_metadata.json
    └── insights.json
```

---

## How Claude Integration Works

### Claude Client (`utils/claude_client.py`)

Simple wrapper around the Anthropic API:

```python
from utils.claude_client import claude

# Make a call
response = claude.call(
    prompt="Analyze this data: {...}",
    system_prompt="You are a data quality expert",
    max_tokens=2048
)
```

**Under the hood:**
1. Uses Anthropic Python SDK
2. Reads `CLAUDE_API_KEY` from environment
3. Uses model: `claude-sonnet-4-5-20250929`
4. Returns text response

### Best Practices in Your Implementation

1. **Structured Prompts:** Always ask Claude to return JSON
2. **Clean JSON Responses:** Strip markdown code blocks (```json)
3. **Error Handling:** Try-except with fallbacks
4. **Clear Instructions:** "Return ONLY valid JSON, no explanations"

Example from your code:
```python
prompt = f"""Analyze this dataset schema.

{json.dumps(data, indent=2)}

Return JSON with this structure:
{{
  "columns": [...],
  "warnings": [...]
}}

Return ONLY valid JSON."""

response = claude.call(prompt, max_tokens=2048)

# Clean response
response_clean = response.strip()
if response_clean.startswith("```"):
    # Strip code blocks
    lines = response_clean.split('\n')
    response_clean = '\n'.join(lines[1:-1])

result = json.loads(response_clean)
```

---

## Configuration (`config/config.py`)

All paths and settings in one place:

```python
# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# Files
SCHEMA_FILE = OUTPUT_DIR / "schema.json"
TRANSFORMATION_LOG_FILE = OUTPUT_DIR / "transformation_log.json"
PLOT_METADATA_FILE = OUTPUT_DIR / "plot_metadata.json"
INSIGHTS_FILE = OUTPUT_DIR / "insights.json"
STATUS_FILE = LOGS_DIR / "status.json"

# Claude API
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_RETRIES = 3
```

---

## Data Flow Example

Let's trace a file through the entire system:

### Input: `test_data.csv`
```csv
name,age,salary,department,join_date
Alice,25,50000,Engineering,2020-01-15
Bob,30,60000,Marketing,2019-03-20
Charlie,35,70000,Engineering,2018-06-10
Diana,28,55000,Sales,2021-02-05
Eve,32,65000,Engineering,2017-11-30
```

### Stage 1: Data Agent

**1. Ingest:** Detects CSV, 320 bytes
**2. Preprocess:** Loads into DataFrame (5 rows × 5 columns)
**3. Infer Schema (Claude):**
   - Identifies `age` as numeric (mean: 30, range: 25-35)
   - Identifies `department` as categorical (3 unique: Engineering, Marketing, Sales)
   - Identifies `join_date` as datetime string
   - **Issues:** join_date should be datetime type
   - **Warnings:** Small dataset, Engineering 60% (imbalanced)

**4. Clean Data (Claude):**
   - Operation 1: Normalize `department` (lowercase, trim)
   - Operation 2: Normalize `name` (lowercase, trim)
   - No rows dropped

**5. Generate Report (Claude):**
   - Quality Score: 7.5/10
   - Highlights: Good completeness, needs datetime conversion
   - Recommendations: Convert join_date, verify sample size

**Outputs:**
- `cleaned_data.csv` - Same data, normalized
- `schema.json` - Rich schema with Claude insights
- `transformation_log.json` - 2 operations logged
- `data_quality_report.md` - Professional report

### Stage 2: Plot Agent (Placeholder)

**Inputs:** cleaned_data.csv, schema.json
**Process:** Creates basic histogram (placeholder)
**Outputs:**
- `plots/histogram_age.png`
- `plot_metadata.json`

### Stage 3: Analysis Agent (Placeholder)

**Inputs:** All previous outputs
**Process:** Placeholder analysis
**Outputs:**
- `insights.json` (empty)
- `analysis_report.md` (basic)

---

## Testing Your Work

### Test 1: Basic Pipeline
```bash
python main.py test_data.csv
```
Expected: All 3 stages complete, files created

### Test 2: Check Claude Integration
```bash
# Should see "✓ Schema inferred" not "⚠️ Claude failed"
python main.py test_data.csv | grep "Schema inferred"

# View the intelligent schema
cat output/schema.json | jq '.columns[] | {name, inferred_type, semantic_meaning}'
```

### Test 3: Verify Outputs
```bash
# Check all files exist
ls -lh output/schema.json
ls -lh output/cleaned/cleaned_data.csv
ls -lh output/transformation_log.json
ls -lh output/reports/data_quality_report.md

# View the report
cat output/reports/data_quality_report.md
```

### Test 4: Check Cleaning Operations
```bash
# See what transformations Claude recommended
cat output/transformation_log.json | jq '.operations'
```

### Test 5: Test Without API Key
```bash
# Temporarily rename .env
mv .env .env.backup

# Run - should use fallbacks
python main.py test_data.csv
# Expected: "⚠️ Claude ... failed" but pipeline still completes

# Restore .env
mv .env.backup .env
```

---

## Troubleshooting

### "Could not resolve authentication method"
- **Cause:** No CLAUDE_API_KEY in .env
- **Fix:** `echo "CLAUDE_API_KEY=your-key" > .env`

### "ModuleNotFoundError: anthropic"
- **Cause:** Dependencies not installed
- **Fix:** `source activate.sh` or `pip install -r requirements.txt`

### Pipeline runs but no Claude analysis
- **Check:** Look for "⚠️ Claude ... failed" in output
- **Debug:** Run with key visible: `echo $CLAUDE_API_KEY | head -c 20`
- **Test:** Try calling Claude directly:
  ```python
  from utils.claude_client import claude
  print(claude.call("Say hello"))
  ```

### Outputs are empty/placeholder
- **Data Agent:** Check if "✓ Schema inferred" appears (vs "⚠️ ... failed")
- **Plot/Analysis:** These are still TODO for Nikunj/Shamanth

---

## What's Complete vs TODO

### ✅ Complete (Your Work)

- [x] CEO Orchestrator with full pipeline
- [x] Data Agent with Claude-powered:
  - [x] Schema inference (semantic types, quality analysis)
  - [x] Smart data cleaning (6 operation types)
  - [x] Professional quality reports
- [x] Graceful fallbacks for all Claude calls
- [x] Status tracking and logging
- [x] Error handling and retries

### 🚧 TODO (Teammates)

- [ ] **Plot Agent (Nikunj):**
  - [ ] Claude-powered plot planning
  - [ ] Generate multiple plot types
  - [ ] Add plot descriptions

- [ ] **Analysis Agent (Shamanth):**
  - [ ] Statistical analysis with Claude
  - [ ] Plot interpretation
  - [ ] Business insights generation

- [ ] **UI (Amit):**
  - [ ] File upload with preview
  - [ ] Real-time status polling
  - [ ] Display all outputs
  - [ ] Retry buttons

---

## Key Takeaways

1. **CEO Pattern:** All actions go through CEO - enables retry, logging, orchestration
2. **Claude Integration:** Each agent uses Claude for intelligent decisions, with fallbacks
3. **File-based Communication:** Agents communicate via files, not direct calls
4. **Modular Design:** Each agent is independent, can be developed in parallel
5. **Production-Ready:** Error handling, logging, retries all built-in

Your Data Agent is the **most critical** component - it ensures data quality before visualization and analysis. The Claude-powered features you built provide real intelligence that would be hard to replicate with rules alone.
