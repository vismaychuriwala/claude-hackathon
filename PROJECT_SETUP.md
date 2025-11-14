# Multi-Agent Data Command Center - Project Setup

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variable
export CLAUDE_API_KEY="your-api-key-here"
```

### 2. Run the Application

```bash
# Option 1: Start web UI
python main.py ui

# Option 2: Run pipeline on a file directly
python main.py path/to/your/data.csv
```

---

## Project Structure

```
claude-hackathon/
├── config/
│   └── config.py              # Configuration (paths, API settings)
├── core/
│   └── ceo.py                 # CEO Orchestrator (routes all requests)
├── agents/
│   ├── __init__.py
│   ├── data_agent.py          # VISMAY - Data pipeline
│   ├── plot_agent.py          # NIKUNJ - Visualizations
│   └── analysis_agent.py      # SHAMANTH - Analysis & insights
├── ui/
│   ├── app.py                 # AMIT - Flask backend
│   └── templates/
│       └── index.html         # AMIT - Frontend HTML
├── utils/
│   └── claude_client.py       # Claude API wrapper
├── output/                    # All outputs written here
│   ├── raw/                   # Uploaded files
│   ├── cleaned/               # Cleaned data
│   ├── plots/                 # Generated plots
│   ├── reports/               # Reports (MD files)
│   └── logs/                  # Status logs
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── TEAM_TODO.md              # Detailed task breakdown
└── PROJECT_SETUP.md          # This file
```

---

## Team Assignments

### **VISMAY** (Person 2) - Data Pipeline Agent
**File**: `agents/data_agent.py`

**TODO Sections**:
1. `_ingest_file()` - Detect file type (CSV, Excel, PDF, JSON)
2. `_preprocess()` - Convert to DataFrame, handle OCR for PDFs
3. `_infer_schema()` - **Use Claude** to infer column types intelligently
4. `_clean_data()` - **Use Claude** to generate cleaning strategy
5. `_generate_report()` - **Use Claude** to create quality report

**Outputs**:
- `output/cleaned/cleaned_data.csv`
- `schema.json`
- `transformation_log.json`
- `reports/data_quality_report.md`

**How to use Claude**:
```python
from utils.claude_client import claude

# Example: Infer schema
prompt = f"""
Analyze this DataFrame sample and infer the schema:
{df.head(10).to_string()}

For each column, determine:
- Type (numeric, categorical, datetime, text, geo)
- Null percentage
- Warnings (data quality issues)

Return JSON format.
"""
response = claude.call(prompt)
schema = json.loads(response)
```

---

### **NIKUNJ** (Person 3) - Visualization Agent
**File**: `agents/plot_agent.py`

**TODO Sections**:
1. `_load_data_and_schema()` - Load cleaned CSV and schema.json
2. `_plan_plots()` - **Use Claude** to decide what plots to create
3. `_generate_plot()` - Create plots with matplotlib/plotly, **use Claude** for descriptions

**Outputs**:
- `output/plots/*.png` - Plot images
- `output/plots/*.html` - Interactive plots (optional)
- `plot_metadata.json`

**Plot Types**:
- Histograms for numeric columns
- Bar charts for categorical
- Correlation heatmap
- Missingness heatmap
- Time series (if datetime columns)

**How to use Claude**:
```python
from utils.claude_client import claude

# Example: Plan plots
prompt = f"""
Given this schema:
{json.dumps(schema, indent=2)}

Suggest what plots to create. Return a JSON list of plot plans:
[
  {{"type": "histogram", "column": "age", "title": "..."}},
  {{"type": "bar", "column": "category", "title": "..."}}
]
"""
response = claude.call(prompt)
plot_plan = json.loads(response)
```

---

### **SHAMANTH** (Person 4) - Analysis Agent
**File**: `agents/analysis_agent.py`

**TODO Sections**:
1. `_load_inputs()` - Load all files (data, schema, plots, transform log)
2. `_statistical_analysis()` - **Use Claude** for smart stats analysis
3. `_interpret_plots()` - **Use Claude vision API** to interpret plot images
4. `_data_quality_insights()` - Generate insights about data quality
5. `_business_insights()` - **Use Claude** for actionable business insights
6. `_generate_recommendations()` - **Use Claude** to suggest next steps
7. `_generate_report()` - **Use Claude** to create comprehensive report

**Outputs**:
- `insights.json`
- `reports/analysis_report.md`

**How to use Claude**:
```python
from utils.claude_client import claude

# Example: Generate insights
prompt = f"""
Analyze this dataset statistics:
- Rows: {len(df)}
- Columns: {df.columns.tolist()}
- Summary stats: {df.describe().to_string()}

Generate 3-5 actionable insights in JSON format:
[
  {{
    "type": "anomaly|trend|correlation|quality",
    "severity": "high|medium|low",
    "description": "...",
    "affected_columns": [...],
    "recommendation": "..."
  }}
]
"""
response = claude.call(prompt)
insights = json.loads(response)
```

**Vision API for plots**:
```python
# TODO: Use Claude vision API to analyze plot images
# This requires the Anthropic Messages API with image support
```

---

### **AMIT** (Person 1) - UI Developer
**Files**: `ui/app.py`, `ui/templates/index.html`

**TODO Sections in app.py**:
1. Complete `/api/upload` - Trigger CEO pipeline after upload
2. Implement `/api/retry/<stage>` - Retry failed stages

**TODO Sections in index.html**:
1. Improve styling (add Tailwind CSS or custom CSS)
2. Implement JavaScript functions:
   - `loadSchema()` - Fetch and display schema
   - `loadQualityReport()` - Fetch and display report
   - `loadPlots()` - Fetch plot metadata and display images
   - `loadInsights()` - Fetch and display insights
   - `retryStage(stageName)` - Call retry API
3. Add loading indicators
4. Complete status polling logic
5. Add retry buttons to failed stages

**API Endpoints Available**:
- `POST /api/upload` - Upload file
- `GET /api/status` - Get pipeline status
- `GET /api/schema` - Get schema.json
- `GET /api/plots` - Get plot_metadata.json
- `GET /api/insights` - Get insights.json
- `GET /api/reports/quality` - Get quality report (markdown)
- `GET /api/reports/analysis` - Get analysis report (markdown)
- `GET /api/transformation_log` - Get transformation log
- `GET /plots/<filename>` - Serve plot images
- `POST /api/retry/<stage>` - Retry a stage

**Frontend Framework Suggestions**:
- Vanilla JS (current)
- OR add React/Vue for better state management
- Add Tailwind CSS for styling
- Use marked.js for rendering markdown reports

---

## Data Flow

```
1. User uploads file via UI
   └─> Saved to output/raw/

2. CEO routes to Data Agent
   └─> Processes file
   └─> Writes: cleaned_data.csv, schema.json, transformation_log.json, report.md

3. CEO routes to Plot Agent
   └─> Reads: cleaned_data.csv, schema.json
   └─> Writes: plots/*.png, plot_metadata.json

4. CEO routes to Analysis Agent
   └─> Reads: all previous outputs
   └─> Writes: insights.json, analysis_report.md

5. UI displays all outputs
   └─> Polls status.json for real-time updates
   └─> Loads and displays all files
```

---

## File Format Specifications

### `schema.json`
```json
{
  "columns": [
    {
      "name": "column_name",
      "type": "numeric|categorical|datetime|text|geo",
      "null_pct": 0.05,
      "unique_count": 100,
      "min": 0,
      "max": 1000
    }
  ],
  "warnings": ["Warning message 1", "Warning message 2"]
}
```

### `transformation_log.json`
```json
{
  "operations": [
    {
      "step": 1,
      "operation": "cast_column|impute_missing|remove_duplicates|...",
      "column": "column_name",
      "rows_affected": 10,
      "details": "..."
    }
  ],
  "provenance": {
    "original_rows": 1000,
    "final_rows": 987,
    "columns_dropped": []
  }
}
```

### `plot_metadata.json`
```json
{
  "plots": [
    {
      "filename": "histogram_age.png",
      "html_filename": "histogram_age.html",
      "type": "histogram",
      "column": "age",
      "description": "Distribution shows...",
      "insights": ["Right-skewed", "3 outliers"]
    }
  ],
  "total_plots": 5
}
```

### `insights.json`
```json
{
  "insights": [
    {
      "type": "anomaly|trend|correlation|quality",
      "severity": "high|medium|low",
      "description": "Human readable insight",
      "affected_columns": ["col1", "col2"],
      "recommendation": "Suggested action"
    }
  ],
  "summary": {
    "total_insights": 5,
    "high_severity": 1,
    "recommended_next_steps": ["Step 1", "Step 2"]
  }
}
```

### `status.json`
```json
{
  "current_stage": "plotting",
  "stages": {
    "data": {
      "status": "completed",
      "timestamp": "2025-01-14T10:30:00",
      "message": "Cleaned 1000 rows"
    },
    "plot": {
      "status": "in_progress",
      "timestamp": "2025-01-14T10:31:00",
      "message": "Creating plots..."
    }
  },
  "errors": [],
  "retry_count": 0
}
```

---

## Testing

### Test with Sample CSV

Create a test file `test_data.csv`:
```csv
name,age,salary,department,join_date
Alice,25,50000,Engineering,2020-01-15
Bob,30,60000,Marketing,2019-03-20
Charlie,35,70000,Engineering,2018-06-10
```

Run:
```bash
python main.py test_data.csv
```

---

## Debugging Tips

### Check Status
```bash
# View current status
cat output/logs/status.json | python -m json.tool

# View schema
cat schema.json | python -m json.tool

# View insights
cat insights.json | python -m json.tool
```

### Common Issues

1. **Claude API errors**: Check `CLAUDE_API_KEY` is set
2. **File not found**: Check paths in `config/config.py`
3. **Import errors**: Make sure you're running from project root

---

## Development Workflow

### Hour 1: Setup & CEO Integration
- All: Install dependencies, set up API key
- All: Test basic agent registration with CEO
- All: Implement execute() method stub

### Hours 2-4: Parallel Development
- **Vismay**: Implement data pipeline with Claude
- **Nikunj**: Implement plotting with Claude
- **Shamanth**: Implement analysis with Claude
- **Amit**: Build UI with polling and display

### Hour 5: Integration
- Test full pipeline with sample data
- Fix integration issues
- Test retry mechanism

### Hour 6: Polish & Demo
- Improve UI styling
- Add better error handling
- Prepare demo with real dataset
- Create presentation

---

## Demo Checklist

- [ ] Upload CSV file via UI
- [ ] Show real-time status updates
- [ ] Display inferred schema
- [ ] Show data quality report
- [ ] Display 3-5 plots
- [ ] Show 3-5 AI-generated insights
- [ ] Demonstrate retry button on failed stage
- [ ] Show analysis report

---

## Resources

- **Claude API Docs**: https://docs.anthropic.com/
- **Flask Docs**: https://flask.palletsprojects.com/
- **Pandas Docs**: https://pandas.pydata.org/
- **Matplotlib Docs**: https://matplotlib.org/
- **Plotly Docs**: https://plotly.com/python/

---

## Contact

- **Vismay** (You): Data Agent
- **Nikunj**: Plot Agent
- **Shamanth**: Analysis Agent
- **Amit**: UI

**Share this file with your team!**
