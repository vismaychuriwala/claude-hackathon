# Quick Start Guide

## ✅ YES - It's Runnable Now!

The project runs end-to-end with placeholder implementations. All TODO sections just print messages and create basic outputs.

## 🚀 Installation (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set Claude API key (not needed for testing structure)
export CLAUDE_API_KEY="your-key-here"
```

## 🎯 Run Commands

### Option 1: Run Pipeline on CSV
```bash
python main.py test_data.csv
```

**What happens**:
- ✓ All 3 agents run in sequence
- ✓ Creates placeholder outputs (CSV, JSON, plots, reports)
- ✓ Shows all TODO markers in console
- ✓ Pipeline completes successfully

### Option 2: Start Web UI
```bash
python main.py ui
```

**What happens**:
- ✓ Flask server starts on http://localhost:5000
- ✓ UI loads with file upload, status panels
- ✓ Can upload files (triggers placeholder pipeline)
- ✓ Displays generated outputs

### Option 3: Just See Usage
```bash
python main.py
```

Shows help and team assignments.

---

## 📊 What Gets Created (Placeholders)

When you run `python main.py test_data.csv`:

```
output/
├── cleaned/
│   └── cleaned_data.csv        ✓ Copy of input (placeholder)
├── plots/
│   └── histogram_age.png       ✓ Basic matplotlib plot
├── reports/
│   ├── data_quality_report.md  ✓ Placeholder report
│   └── analysis_report.md      ✓ Placeholder report
├── logs/
│   └── status.json             ✓ Pipeline status
├── schema.json                 ✓ Basic pandas dtypes
├── transformation_log.json     ✓ Empty operations list
├── plot_metadata.json          ✓ Plot info
└── insights.json               ✓ Empty insights
```

---

## 👀 What You'll See

### Console Output (with TODOs):
```
[DataAgent] TODO: Implement _ingest_file()
[DataAgent] TODO: Implement _preprocess()
[DataAgent] TODO: Implement _infer_schema() using Claude
[DataAgent] TODO: Implement _clean_data() using Claude
✓ Data processed: output/cleaned/cleaned_data.csv

[PlotAgent] TODO: Implement _plan_plots() using Claude
✓ Created 1 plots

[AnalysisAgent] TODO: Implement _statistical_analysis() using Claude
✓ Generated 0 insights

✓ Pipeline complete!
```

### Files Created:
All JSON files are valid (can be parsed)
All CSV files contain data
All MD files have placeholder text
Plots are real PNG images (basic histogram)

---

## 🔧 What Works (Placeholders)

| Component | Status | Output |
|-----------|--------|--------|
| **CEO Orchestrator** | ✅ Working | Routes requests, handles retries, logs status |
| **Data Agent** | ✅ Runnable | Reads CSV, creates basic schema, saves outputs |
| **Plot Agent** | ✅ Runnable | Creates 1 histogram with matplotlib |
| **Analysis Agent** | ✅ Runnable | Creates empty insights JSON |
| **Flask UI** | ✅ Working | Serves on port 5000, has all endpoints |
| **API Endpoints** | ✅ Working | Return JSON/files (from placeholders) |

---

## 🎯 Next Steps (Fill in TODOs)

### For You (Vismay) - `agents/data_agent.py`:
```python
def _infer_schema(self, df):
    # TODO: Replace this placeholder with Claude call
    prompt = f"Infer schema from: {df.head().to_string()}"
    response = claude.call(prompt)
    return json.loads(response)
```

### For Nikunj - `agents/plot_agent.py`:
```python
def _plan_plots(self, df, schema):
    # TODO: Use Claude to plan plots intelligently
    prompt = f"What plots for schema: {schema}"
    response = claude.call(prompt)
    return json.loads(response)
```

### For Shamanth - `agents/analysis_agent.py`:
```python
def _business_insights(self, df, schema, stats):
    # TODO: Use Claude for insights
    prompt = f"Generate insights from: {stats}"
    response = claude.call(prompt)
    return json.loads(response)
```

### For Amit - `ui/templates/index.html`:
```javascript
// TODO: Implement these functions
function loadSchema() { /* fetch /api/schema */ }
function loadPlots() { /* fetch /api/plots */ }
function loadInsights() { /* fetch /api/insights */ }
```

---

## ✅ Verify Installation

```bash
# Should show usage
python main.py

# Should create outputs
python main.py test_data.csv

# Should start server
python main.py ui
# Then visit: http://localhost:5000
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: anthropic"
```bash
pip install anthropic
```

### "No module named 'agents'"
```bash
# Make sure you're in the project root
cd /home/vismay/claude-hackathon
python main.py
```

### UI not loading
- Check Flask started: Should see "Running on http://127.0.0.1:5000"
- Visit: http://localhost:5000
- Check firewall/port 5000

---

## 📝 Summary

**YES - Fully runnable!**

The structure is complete with:
- ✅ All imports work
- ✅ All agents registered
- ✅ Pipeline runs end-to-end
- ✅ Files get created
- ✅ UI starts and serves pages
- ✅ No runtime errors

The TODOs are **clearly marked** for each person to fill in with actual Claude-powered logic. Right now it's all placeholders, but the **plumbing works perfectly**.

**You can start working on your TODOs immediately!**
