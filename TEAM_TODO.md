# Multi-Agent Data Command Center - Team Task Breakdown

**Project**: Agentic AI + MCP Data Extraction Pipeline
**Team Size**: 4 people working in parallel
**Architecture**: CEO Orchestrator pattern with feedback loops

---

## Architecture Overview

```
User Upload → CEO Orchestrator
                  ↓
              [Request Queue]
                  ↓
         ┌────────┼────────┬─────────┐
         ↓        ↓        ↓         ↓
    IngestAgent DataAgent PlotAgent AnalysisAgent
         ↓        ↓        ↓         ↓
    [Write outputs to filesystem]
         ↓        ↓        ↓         ↓
    [Send response to CEO: SUCCESS | FAILURE | RETRY_NEEDED]
         ↓
    CEO logs status.json
         ↓
    UI polls & displays
         ↓
    User clicks "Retry" → CEO re-queues failed task
```

**Key Principle**: ALL agent actions go through CEO. Most actions write to files for UI display.

---

## SHARED INFRASTRUCTURE: CEO ORCHESTRATOR

**Responsibility**: Central coordinator that routes all agent actions and handles retries

### Tasks (5)

- [ ] **CEO-1**: Create CEO agent class with request queue, agent registry, and retry mechanism (max 3 retries per task)
- [ ] **CEO-2**: Implement status/event logging system (writes to `status.json`) for UI consumption
- [ ] **CEO-3**: Build agent communication protocol (request/response format with success/failure/retry states)
- [ ] **CEO-4**: Create workflow orchestration logic (Ingest → Data → Plotting → Analysis pipeline)
- [ ] **CEO-5**: Set up shared `output/` directory structure: `raw/`, `cleaned/`, `plots/`, `reports/`, `logs/`

### Output Files
- `status.json` - Real-time agent status and logs

---

## PERSON 1: UI DEVELOPER

**Responsibility**: Build dashboard that displays all agent outputs and provides user controls

### Tasks (9)

- [ ] **UI-1**: Set up web framework (Flask/FastAPI backend + React/Vue frontend) with file upload endpoint
- [ ] **UI-2**: Create file upload component with drag-drop and file type preview
- [ ] **UI-3**: Build real-time status dashboard that polls `status.json` and shows agent progress/logs
- [ ] **UI-4**: Create data preview panel that displays raw data sample and inferred schema from `schema.json`
- [ ] **UI-5**: Build data quality report viewer that renders `data_quality_report.md` with warnings highlighted
- [ ] **UI-6**: Create visualization gallery that loads and displays all plots from `output/plots/` directory
- [ ] **UI-7**: Build analysis insights panel that displays `insights.json` and `analysis_report.md`
- [ ] **UI-8**: Add retry buttons for each agent stage (request retry through CEO API endpoint)
- [ ] **UI-9**: Implement transformation log viewer showing step-by-step data changes from `transformation_log.json`

### Files Read by UI
- `status.json` - Agent status and logs
- `schema.json` - Inferred data schema
- `data_quality_report.md` - Data quality summary
- `transformation_log.json` - Data transformation steps
- `plot_metadata.json` - Plot descriptions
- `output/plots/*.png` - Plot images
- `output/plots/*.html` - Interactive plots
- `insights.json` - Structured insights
- `analysis_report.md` - Analysis summary

### Parallel Work Streams
1. File upload + preview (independent)
2. Status dashboard (polls CEO status)
3. Data panels (schema, quality, transformations)
4. Visualization gallery
5. Analysis/insights display

---

## PERSON 2: DATA PIPELINE ENGINEER

**Responsibility**: Build data ingestion, preprocessing, schema inference, and cleaning agents

### Agents to Build
- **IngestAgent** - File type detection
- **PreprocessorAgent** - Format conversion
- **SchemaAgent** - Type inference
- **CleanerAgent** - Data cleaning

### Tasks (11)

- [ ] **DATA-1**: Create IngestAgent that detects file types (CSV/Excel/PDF/JSON/images/HTML) using magic bytes + extensions
- [ ] **DATA-2**: Build PreprocessorAgent for format conversion (PDF→text via OCR, HTML→tables, Excel→CSV, images→text)
- [ ] **DATA-3**: Implement SchemaAgent that infers column types (numeric/categorical/datetime/text/geo) and computes stats
- [ ] **DATA-4**: Write schema inference output to `schema.json` with columns, types, null_pct, warnings
- [ ] **DATA-5**: Build CleanerAgent that applies transformations (type casting, datetime standardization, deduplication)
- [ ] **DATA-6**: Implement missing value handling (drop if null>50%, else impute with mean/median/sentinel)
- [ ] **DATA-7**: Add outlier detection (IQR/z-score) and categorical normalization (lowercase, synonym mapping)
- [ ] **DATA-8**: Write cleaned data to `output/cleaned/cleaned_data.csv` or `cleaned_data.parquet`
- [ ] **DATA-9**: Generate `transformation_log.json` with list of operations, row counts affected, and provenance
- [ ] **DATA-10**: Generate `data_quality_report.md` with summary, warnings, null percentages, type issues
- [ ] **DATA-11**: Integrate with CEO - register agents, handle requests, send success/failure responses with retry requests

### Files Written
- `output/raw/original_file.*` - Original uploaded file
- `output/cleaned/cleaned_data.csv` - Cleaned dataset
- `schema.json` - Inferred schema with types, nulls, warnings
- `transformation_log.json` - Operations log with provenance
- `data_quality_report.md` - Human-readable quality report

### CEO Integration
- Send **SUCCESS** when data cleaned successfully
- Send **FAILURE** if file unreadable or corrupted
- Request **RETRY** if schema ambiguous or user clarification needed

### Parallel Work Streams
1. IngestAgent (file detection) - independent
2. PreprocessorAgent (format conversion) - independent
3. SchemaAgent (type inference) - depends on preprocessor
4. CleanerAgent (transformations) - depends on schema

---

## PERSON 3: VISUALIZATION ENGINEER

**Responsibility**: Build plotting agent that creates charts from cleaned data

### Agent to Build
- **PlottingAgent** - Automatic plot generation

### Tasks (9)

- [ ] **VIZ-1**: Create PlottingAgent that reads `cleaned_data.csv` and `schema.json` to determine plot types
- [ ] **VIZ-2**: Implement automatic plot selection logic (histograms for numeric, bar charts for categorical, time series for datetime)
- [ ] **VIZ-3**: Generate distribution plots for numeric columns (histograms with outliers marked)
- [ ] **VIZ-4**: Generate categorical analysis plots (top N categories bar chart, frequency distributions)
- [ ] **VIZ-5**: Generate correlation heatmap for numeric columns and missingness heatmap
- [ ] **VIZ-6**: Generate time series plots if datetime columns exist (trends, seasonality patterns)
- [ ] **VIZ-7**: Save all plots to `output/plots/` directory as PNG and interactive HTML (using plotly/matplotlib)
- [ ] **VIZ-8**: Generate `plot_metadata.json` with plot filenames, types, descriptions, and column mappings
- [ ] **VIZ-9**: Integrate with CEO - register agent, handle plot requests, send retry requests if data quality insufficient

### Files Read
- `output/cleaned/cleaned_data.csv` - Cleaned dataset
- `schema.json` - Column types and metadata

### Files Written
- `output/plots/histogram_<column>.png` - Distribution plots
- `output/plots/histogram_<column>.html` - Interactive distributions
- `output/plots/bar_<column>.png` - Categorical charts
- `output/plots/correlation_heatmap.png` - Correlation matrix
- `output/plots/missingness_heatmap.png` - Missing data visualization
- `output/plots/timeseries_<column>.png` - Time series plots
- `plot_metadata.json` - Plot catalog with descriptions

### CEO Integration
- Send **SUCCESS** when all plots generated
- Send **FAILURE** if data file missing or corrupted
- Request **RETRY** if data quality insufficient or schema unclear

### Parallel Work Streams
1. Distribution plots (histograms) - independent
2. Categorical plots (bar charts) - independent
3. Heatmaps (correlation, missingness) - independent
4. Time series plots - independent

---

## PERSON 4: ANALYSIS ENGINEER

**Responsibility**: Build analysis agent that interprets plots and data to generate insights

### Agent to Build
- **AnalysisAgent** - Statistical analysis and insight generation

### Tasks (9)

- [ ] **ANALYSIS-1**: Create AnalysisAgent that reads `cleaned_data.csv`, `schema.json`, `plot_metadata.json`, and `transformation_log.json`
- [ ] **ANALYSIS-2**: Implement statistical analysis (compute mean/median/std, detect skewness, identify anomalies)
- [ ] **ANALYSIS-3**: Build plot interpretation logic (analyze histogram shapes, identify outliers, detect patterns in time series)
- [ ] **ANALYSIS-4**: Generate data quality insights (missing data hotspots, transformation impact, data issues)
- [ ] **ANALYSIS-5**: Generate business insights (top anomalies, key trends, interesting correlations, recommended actions)
- [ ] **ANALYSIS-6**: Create recommendation engine (suggest next models, identify key features, propose additional cleaning)
- [ ] **ANALYSIS-7**: Write `insights.json` with structured insights (type, severity, description, affected_columns)
- [ ] **ANALYSIS-8**: Generate `analysis_report.md` with executive summary, key findings, visualizations referenced, recommendations
- [ ] **ANALYSIS-9**: Integrate with CEO - register agent, handle analysis requests, request retry if plots missing or data unclear

### Files Read
- `output/cleaned/cleaned_data.csv` - Cleaned dataset
- `schema.json` - Column metadata
- `plot_metadata.json` - Available plots
- `transformation_log.json` - Data transformations applied
- `output/plots/*` - Plot files (for interpretation)

### Files Written
- `insights.json` - Structured insights
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
    ]
  }
  ```
- `analysis_report.md` - Executive summary with key findings and recommendations

### CEO Integration
- Send **SUCCESS** when analysis complete
- Send **FAILURE** if required files missing
- Request **RETRY** if plots missing or data unclear

### Parallel Work Streams
1. Statistical analysis (stats computation) - independent
2. Plot interpretation (shape analysis) - depends on plots
3. Data quality insights - independent
4. Business insights (recommendations) - depends on stats + plots

---

## INTEGRATION & TESTING

### Shared Tasks (6)

- [ ] **INT-1**: Create shared `output/` directory structure (`raw/`, `cleaned/`, `plots/`, `reports/`, `logs/`)
- [ ] **INT-2**: Define standard file formats and naming conventions for all agent outputs
- [ ] **INT-3**: Test CSV upload → full pipeline → UI display with sample dataset
- [ ] **INT-4**: Test retry mechanism - trigger Data Agent failure and verify CEO retry + UI notification
- [ ] **INT-5**: Test PDF with table extraction → cleaning → plotting → analysis workflow
- [ ] **INT-6**: Verify all outputs are written correctly and UI can read/display them in real-time

---

## Data Flow & File Dependencies

```
1. User uploads file
   └─> IngestAgent → writes output/raw/original_file.*

2. PreprocessorAgent reads raw file
   └─> writes output/raw/preprocessed.*

3. SchemaAgent reads preprocessed file
   └─> writes schema.json

4. CleanerAgent reads preprocessed + schema.json
   └─> writes output/cleaned/cleaned_data.csv
   └─> writes transformation_log.json
   └─> writes data_quality_report.md

5. PlottingAgent reads cleaned_data.csv + schema.json
   └─> writes output/plots/*.png, *.html
   └─> writes plot_metadata.json

6. AnalysisAgent reads cleaned_data.csv + schema.json + plot_metadata.json + transformation_log.json
   └─> writes insights.json
   └─> writes analysis_report.md

7. UI reads ALL output files and displays them
```

---

## Retry/Feedback Scenarios

### Data Agent Retry Scenarios
- **File unreadable**: Request retry with error details, suggest file format
- **Schema ambiguous**: Request user clarification on column types
- **Too many nulls**: Warn user, request confirmation to proceed

### Plotting Agent Retry Scenarios
- **Data quality insufficient**: Request Data Agent re-clean with stricter rules
- **Missing required columns**: Request schema clarification
- **Unable to infer plot type**: Request user preference

### Analysis Agent Retry Scenarios
- **Plots missing**: Request Plotting Agent re-run
- **Data unclear**: Request Data Agent for additional stats
- **Insufficient information**: Request user context (e.g., "What is this dataset for?")

---

## Quick Start Checklist

### Before You Start
- [ ] Clone/create project repository
- [ ] Set up virtual environment (Python 3.9+)
- [ ] Install dependencies: `pandas`, `numpy`, `matplotlib`, `plotly`, `scikit-learn`, `flask/fastapi`
- [ ] Create `output/` directory structure

### Coordination Points
1. **Hour 1**: Everyone builds CEO integration first (register agents, send/receive messages)
2. **Hour 2-4**: Parallel development on individual agents
3. **Hour 5**: Integration testing
4. **Hour 6**: Bug fixes + demo prep

### Communication
- Share `schema.json` format early (Person 2 → Person 3, 4)
- Share `plot_metadata.json` format early (Person 3 → Person 4)
- Share `status.json` format early (CEO → Person 1)

---

## Success Criteria

### MVP Demo Flow
1. Upload CSV file (1000 rows)
2. Watch status dashboard show agent progress
3. View inferred schema
4. View data quality report
5. View 3-5 plots (histogram, bar chart, correlation heatmap)
6. View 3-5 insights with recommendations
7. Click "Retry" button on any failed stage

### Bonus Features (if time permits)
- PDF table extraction
- Text column semantic search
- Auto-ML model recommendation
- Interactive plot filtering in UI

---

## File Format Specifications

### `schema.json`
```json
{
  "columns": [
    {
      "name": "order_id",
      "type": "int",
      "null_pct": 0,
      "unique_count": 1000
    },
    {
      "name": "order_date",
      "type": "datetime",
      "format": "%Y-%m-%d",
      "null_pct": 0.02
    },
    {
      "name": "amount",
      "type": "float",
      "null_pct": 0.01,
      "units": "USD",
      "min": 10.5,
      "max": 999.99
    }
  ],
  "warnings": [
    "order_date parsed ambiguously in 3 rows",
    "customer_name has 400 unique values (high cardinality)"
  ]
}
```

### `transformation_log.json`
```json
{
  "operations": [
    {
      "step": 1,
      "operation": "cast_column",
      "column": "order_id",
      "from_type": "object",
      "to_type": "int",
      "rows_affected": 5,
      "errors": 0
    },
    {
      "step": 2,
      "operation": "impute_missing",
      "column": "amount",
      "method": "median",
      "rows_affected": 12
    },
    {
      "step": 3,
      "operation": "remove_duplicates",
      "columns": ["order_id"],
      "rows_removed": 8
    }
  ],
  "provenance": {
    "original_rows": 1000,
    "final_rows": 987,
    "columns_dropped": ["temp_col"],
    "timestamp": "2025-01-14T10:30:00Z"
  }
}
```

### `plot_metadata.json`
```json
{
  "plots": [
    {
      "filename": "histogram_amount.png",
      "html_filename": "histogram_amount.html",
      "type": "histogram",
      "column": "amount",
      "description": "Distribution of order amounts showing right skew with outliers above $800",
      "insights": ["Right-skewed distribution", "3 outliers detected"]
    },
    {
      "filename": "correlation_heatmap.png",
      "type": "heatmap",
      "columns": ["amount", "quantity", "discount"],
      "description": "Correlation between numeric columns"
    }
  ]
}
```

### `insights.json`
```json
{
  "insights": [
    {
      "type": "anomaly",
      "severity": "high",
      "description": "3 orders with amount > $800 detected (3 std deviations above mean)",
      "affected_columns": ["amount"],
      "affected_rows": [45, 234, 789],
      "recommendation": "Review these orders manually or apply capping at 99th percentile"
    },
    {
      "type": "quality",
      "severity": "medium",
      "description": "12% missing values in 'discount' column",
      "affected_columns": ["discount"],
      "recommendation": "Imputed with median (0). Consider if missing means 'no discount'."
    },
    {
      "type": "trend",
      "severity": "low",
      "description": "Order amounts show 15% increase over time period",
      "affected_columns": ["amount", "order_date"],
      "recommendation": "Consider time-based features for forecasting model"
    }
  ],
  "summary": {
    "total_insights": 3,
    "high_severity": 1,
    "recommended_next_steps": [
      "Handle outliers in amount column",
      "Build time-series forecasting model",
      "Investigate missing discount values"
    ]
  }
}
```

### `status.json`
```json
{
  "current_stage": "plotting",
  "stages": {
    "ingest": {
      "status": "completed",
      "timestamp": "2025-01-14T10:25:00Z",
      "message": "Detected CSV file, 1000 rows"
    },
    "preprocess": {
      "status": "completed",
      "timestamp": "2025-01-14T10:25:30Z",
      "message": "Converted to dataframe"
    },
    "schema": {
      "status": "completed",
      "timestamp": "2025-01-14T10:26:00Z",
      "message": "Inferred 8 columns (3 numeric, 2 categorical, 2 datetime, 1 text)"
    },
    "clean": {
      "status": "completed",
      "timestamp": "2025-01-14T10:27:00Z",
      "message": "Cleaned 1000 → 987 rows (13 duplicates removed)"
    },
    "plotting": {
      "status": "in_progress",
      "timestamp": "2025-01-14T10:27:30Z",
      "message": "Generating 5 plots..."
    },
    "analysis": {
      "status": "pending",
      "timestamp": null,
      "message": null
    }
  },
  "errors": [],
  "retry_count": 0
}
```

---

## Questions or Issues?

- Check `status.json` for agent status
- Review `data_quality_report.md` for data issues
- Check `transformation_log.json` for what operations were applied
- Use retry buttons in UI if any stage fails

Good luck! 🚀
