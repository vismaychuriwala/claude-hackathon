# Intelligent Agent System with Sandbox Execution

## Overview

The multi-agent system has been upgraded from using **fixed, hardcoded operations** to **flexible, Claude-directed code generation with sandbox execution**. This solves the "blindly creating mean" problem and enables intelligent, context-aware analysis.

---

## Key Improvements

### Before: Fixed Operation Mode ❌

**Problem**: Agents used predefined operations without understanding data context.

```python
# OLD Analysis Agent - Blindly computes stats for ALL numeric columns
for col in numeric_cols:
    stats[col] = {
        "mean": col_data.mean(),      # Even for ID columns!
        "median": col_data.median(),  # Even for ZIP codes!
        "std": col_data.std()         # Even for categorical encoded as numbers!
    }
```

**Issues**:
- ❌ Computed mean for customer IDs, ZIP codes, phone numbers
- ❌ No validation if statistics are meaningful
- ❌ Claude only interpreted AFTER computation (can't change approach)
- ❌ Limited to hardcoded operations (mean, median, std, etc.)

### After: Intelligent Sandbox Mode ✅

**Solution**: Claude plans analyses FIRST, generates custom code, executes in sandbox, validates results.

```python
# NEW Analysis Agent - Claude-directed intelligent analysis
# Step 1: Claude analyzes dataset and plans appropriate analyses
analysis_plan = intelligent_analyzer.plan_analysis(df, schema)
# Returns: "For customer_id: compute cardinality, NOT mean"
#          "For age: compute mean, median, distribution shape"

# Step 2: Execute Claude-generated code in sandbox
results = intelligent_analyzer.execute_analysis(df, analysis_plan)

# Step 3: Validate results with feedback loop
validation = intelligent_analyzer.validate_results(df, results)
```

**Benefits**:
- ✅ Context-aware: Claude determines appropriate analyses per column
- ✅ Flexible: Can generate custom code beyond predefined operations
- ✅ Safe: Code executes in restricted sandbox environment
- ✅ Validated: Feedback loop ensures quality results

---

## Architecture

### Sandbox Execution System (`utils/sandbox.py`)

**Purpose**: Safely execute agent-generated Python code in isolated environment.

**Features**:
- **Restricted namespace**: No file I/O, no imports, no system calls
- **Timeout enforcement**: 30-60 second limit per execution
- **Output capture**: Captures stdout/stderr for debugging
- **Result validation**: Type checking and custom validation functions

**Allowed modules**: pandas, numpy, matplotlib, seaborn, scipy.stats

**Example Usage**:
```python
from utils.sandbox import execute_analysis_code

code = """
# Claude-generated code
col_data = df['age'].dropna()
analysis_result = {
    'mean': float(col_data.mean()),
    'median': float(col_data.median()),
    'distribution': 'normal' if abs(col_data.skew()) < 0.5 else 'skewed'
}
"""

result = execute_analysis_code(code, df, return_variable="analysis_result")
# result.success = True
# result.result = {'mean': 32.5, 'median': 31.0, 'distribution': 'normal'}
```

---

### Intelligent Analysis System (`utils/intelligent_analysis.py`)

**Components**:

#### 1. DatasetProfile
Comprehensive dataset profiling for intelligent planning:
- Column-level profiling (type, cardinality, distribution)
- Detects likely identifiers (high uniqueness)
- Detects categorical-as-numeric (low unique values)
- Identifies time series, groupings, hierarchies
- Infers business context from column names

#### 2. IntelligentAnalyzer
Main analysis orchestrator:

**Method: `plan_analysis(df, schema)`**
- Sends dataset profile to Claude
- Claude recommends appropriate analyses
- Returns plan with rationale for each analysis

**Example Plan**:
```json
{
  "recommended_analyses": [
    {
      "name": "sales_descriptive_stats",
      "description": "Compute statistics for sales amount",
      "rationale": "Sales is a true quantitative variable suitable for mean/median",
      "code": "analysis_result = {'mean': df['sales'].mean(), ...}",
      "expected_output_type": "dict"
    }
  ],
  "inappropriate_analyses": [
    {
      "name": "mean_of_customer_id",
      "reason": "customer_id is an identifier, not quantitative. Mean is meaningless."
    }
  ]
}
```

**Method: `execute_analysis(df, plan)`**
- Executes each planned analysis in sandbox
- Captures results and execution time
- Handles errors gracefully with detailed logs

**Method: `validate_results(df, profile, results)`**
- Claude validates if results are reasonable
- Identifies suspicious values
- Suggests refinements if needed

---

### Intelligent Data Cleaning (`utils/intelligent_cleaning.py`)

**Purpose**: Generate flexible data cleaning code instead of fixed operations.

**Old Approach**: Select from predefined operations
```python
operations = ["drop_column", "impute_missing", "cast_type", "remove_duplicates",
              "handle_outliers", "normalize_categorical"]
```

**New Approach**: Generate custom cleaning code
```python
cleaner = IntelligentDataCleaner()

# Claude analyzes data quality issues
cleaning_plan = cleaner.plan_cleaning(df, schema)

# Claude generates Python code for cleaning
# Can do anything, not limited to 6 operations!
code = """
# Remove leading zeros from ZIP codes
df['zip_code'] = df['zip_code'].str.lstrip('0')

# Complex business logic
df['revenue_category'] = pd.cut(df['revenue'],
                                bins=[0, 10000, 50000, float('inf')],
                                labels=['small', 'medium', 'large'])
"""

# Execute in sandbox
result = cleaner.execute_cleaning(df, cleaning_plan)
```

**Validation**: Checks for excessive data loss, new missing values, removed columns.

---

### Intelligent Plotting (`utils/intelligent_plotting.py`)

**Purpose**: Generate custom visualizations instead of fixed plot templates.

**Old Approach**: Limited to predefined plot types
- Histogram, Box Plot, Scatter, Bar Chart, Time Series, Heatmap

**New Approach**: Generate custom plotting code
```python
plotter = IntelligentPlotter()

# Claude recommends visualizations based on data characteristics
plot_plan = plotter.plan_plots(df, schema)

# Claude generates matplotlib/seaborn/plotly code
# Can create any visualization, not limited to 6 types!
```

---

## Agent Refactoring

### Analysis Agent (`agents/analysis_agent.py`)

**Key Changes**:

**Old `_statistical_analysis()` method** (Lines 130-250):
- Hardcoded: Compute mean/median/std for ALL numeric columns
- Claude only interprets AFTER computation

**New `_statistical_analysis()` method** (Lines 132-279):
```python
def _statistical_analysis(self, df, schema):
    # Step 1: Let Claude plan appropriate analyses
    analysis_plan = self.intelligent_analyzer.plan_analysis(df, schema)

    # Step 2: Execute in sandbox
    execution_results = self.intelligent_analyzer.execute_analysis(df, analysis_plan)

    # Step 3: Validate results
    validation = self.intelligent_analyzer.validate_results(df, profile, execution_results)

    return {
        "analysis_plan": analysis_plan,
        "execution_results": execution_results,
        "validation": validation,
        "intelligent_mode": True
    }
```

**Fallback**: If intelligent mode fails, falls back to basic statistics (Lines 243-279).

---

### Data Agent (`agents/data_agent.py`)

**Key Changes**:

**Old `_clean_data()` method** (Lines 274-483):
- Limited to 6 predefined cleaning operations
- Fixed logic (e.g., "impute numeric with median")

**New `_clean_data()` method** (Lines 276-396):
```python
def _clean_data(self, df, schema):
    # Step 1: Plan cleaning with Claude
    cleaning_plan = self.intelligent_cleaner.plan_cleaning(df, schema)

    # Step 2: Execute Claude-generated code in sandbox
    execution_result = self.intelligent_cleaner.execute_cleaning(df, cleaning_plan)

    # Step 3: Validate cleaning (check for data loss, integrity)
    validation = self.intelligent_cleaner.validate_cleaning(
        original_df=df,
        cleaned_df=execution_result['cleaned_df'],
        transform_log=execution_result['transform_log']
    )

    return execution_result['cleaned_df'], transform_log
```

**Fallback**: Basic duplicate removal if intelligent mode fails (Lines 379-396).

---

### Plot Agent (`agents/plot_agent.py`)

**Key Changes**:

**Old `create_plots()` method** (Lines 43-80):
- Hardcoded plot templates
- Rule-based plot selection (if numeric → histogram, if categorical → bar chart)

**New `create_plots()` method** (Lines 45-128):
```python
def create_plots(self, cleaned_data_path):
    # Step 1: Plan plots with Claude
    plot_plan = self.intelligent_plotter.plan_plots(df, schema)

    # Step 2: Execute Claude-generated plotting code
    execution_results = self.intelligent_plotter.execute_plots(df, plot_plan)

    return {
        "plots": execution_results['plot_metadata'],
        "intelligent_mode": True
    }
```

**Fallback**: Uses old template-based plotting if intelligent mode fails (Lines 105-128).

---

## How It Works: End-to-End Example

### Scenario: Analyzing Customer Dataset

**Dataset Columns**:
- `customer_id`: Integer (identifier)
- `zip_code`: Integer (geographic code)
- `age`: Integer (true quantitative)
- `purchases`: Integer (count)

### Old System Behavior ❌

```python
# OLD Analysis Agent - Blindly computes stats

stats = {}
for col in ['customer_id', 'zip_code', 'age', 'purchases']:
    stats[col] = {
        'mean': df[col].mean(),      # ❌ Mean of customer_id = nonsense!
        'median': df[col].median(),  # ❌ Median of zip_code = nonsense!
        'std': df[col].std()         # ❌ Std of identifiers = meaningless!
    }

# Result: Garbage statistics that mislead analysts
```

### New System Behavior ✅

```python
# NEW Analysis Agent - Intelligent planning

# Step 1: Dataset Profiling
profile = DatasetProfile(df, schema)
# Detects: customer_id has 99% unique values → likely identifier
#          zip_code has specific format → geographic code
#          age has normal distribution → true quantitative
#          purchases has count semantics → true quantitative

# Step 2: Claude Plans Analyses
analysis_plan = intelligent_analyzer.plan_analysis(df, schema)
# Claude decides:
#   - customer_id: Compute cardinality, NOT mean
#   - zip_code: Compute mode (most common), NOT mean
#   - age: Compute mean, median, distribution shape
#   - purchases: Compute sum, mean, median

# Step 3: Execute Appropriate Analyses
recommended_analyses = [
    {
        "name": "customer_id_cardinality",
        "code": "analysis_result = {'unique_customers': df['customer_id'].nunique()}",
        "rationale": "ID column, only cardinality is meaningful"
    },
    {
        "name": "age_demographics",
        "code": "analysis_result = {'mean_age': df['age'].mean(), 'distribution': ...}",
        "rationale": "Age is quantitative, mean/median are appropriate"
    }
]

# Step 4: Validate Results
validation = intelligent_analyzer.validate_results(...)
# Claude checks: "customer_id mean was NOT computed ✓"
#                "age statistics look reasonable ✓"
```

---

## Configuration

All agents use the intelligent system by default with automatic fallback:

**No configuration required!** The system:
1. Tries intelligent mode first
2. Falls back to basic mode if Claude fails
3. Logs which mode was used in results

**Environment Variables** (optional):
- `CLAUDE_MODEL_DATA_AGENT`: Model for Data Agent (default: haiku-4.5)
- `CLAUDE_MODEL_PLOT_AGENT`: Model for Plot Agent (default: haiku-4.5)
- `CLAUDE_MODEL_ANALYSIS_AGENT`: Model for Analysis Agent (default: haiku-4.5)

---

## Testing

### Test with Sample Dataset

Create `/home/vismay/claude-hackathon/test_intelligent_agents.py`:

```python
import pandas as pd
from utils.intelligent_analysis import IntelligentAnalyzer, DatasetProfile

# Create test dataset with problematic columns
df = pd.DataFrame({
    'customer_id': range(1000, 2000),  # Identifier
    'zip_code': [94102, 10001, 60601] * 333 + [94102],  # Geographic
    'age': [25, 30, 35, 40, 45] * 200,  # True quantitative
    'purchases': [1, 2, 3, 4, 5] * 200  # Count
})

# Run intelligent analysis
analyzer = IntelligentAnalyzer("analysis")
plan = analyzer.plan_analysis(df, schema={})

print("Recommended Analyses:")
for analysis in plan['recommended_analyses']:
    print(f"  - {analysis['name']}: {analysis['rationale']}")

print("\nAvoided Inappropriate Analyses:")
for analysis in plan['inappropriate_analyses']:
    print(f"  - {analysis['name']}: {analysis['reason']}")
```

Run:
```bash
python test_intelligent_agents.py
```

Expected Output:
```
Recommended Analyses:
  - customer_id_cardinality: ID column, only cardinality is meaningful
  - zip_code_mode: Geographic code, mode is most informative
  - age_demographics: Quantitative variable, mean/median appropriate
  - purchases_aggregates: Count variable, sum/mean/median appropriate

Avoided Inappropriate Analyses:
  - mean_of_customer_id: customer_id is an identifier, mean is meaningless
  - mean_of_zip_code: zip_code is geographic, mean is not meaningful
```

---

## Feedback Mechanism

### Single-Pass Validation (Current Implementation)

**Flow**:
1. Plan analyses
2. Execute in sandbox
3. Validate results
4. Return with validation status

**Validation Checks**:
- Are results reasonable given data characteristics?
- Are there suspicious values (e.g., negative ages)?
- Are important analyses missing?

**Example Validation Response**:
```json
{
  "validation_status": "pass",
  "feedback": [
    {
      "analysis": "age_statistics",
      "status": "valid",
      "comment": "Mean age of 32.5 is reasonable for customer dataset"
    },
    {
      "analysis": "income_outliers",
      "status": "suspicious",
      "comment": "Max income of $10M seems unusually high, verify data"
    }
  ],
  "missing_analyses": [
    "Consider segmentation analysis by age groups"
  ]
}
```

### Future: Multi-Iteration Refinement

For multi-iteration feedback (if needed in future):

```python
# Pseudocode for iterative refinement
max_iterations = 3
for i in range(max_iterations):
    plan = analyzer.plan_analysis(df, schema)
    results = analyzer.execute_analysis(df, plan)
    validation = analyzer.validate_results(df, results)

    if validation['status'] == 'pass':
        break  # Success!
    else:
        # Refine plan based on feedback
        schema['feedback'] = validation['suggested_refinements']
```

---

## Benefits Summary

### ✅ Solves "Blindly Creating Mean" Problem
- Claude determines appropriate analyses BEFORE computation
- No more meaningless statistics for identifiers
- Context-aware analysis selection

### ✅ Flexible Code Execution
- Not limited to predefined operations
- Can generate custom analyses/transformations/plots
- Adapts to any dataset structure

### ✅ Safe Sandbox Environment
- Restricted namespace prevents security issues
- Timeout prevents infinite loops
- Error handling with graceful fallbacks

### ✅ Validation Feedback Loop
- Claude validates result quality
- Identifies suspicious values
- Suggests improvements

### ✅ Backward Compatible
- Automatic fallback to basic mode if intelligent mode fails
- Legacy consumers continue to work
- Gradual migration path

---

## Files Modified

1. **New Files Created**:
   - `utils/sandbox.py` - Sandbox execution system
   - `utils/intelligent_analysis.py` - Intelligent analysis planning
   - `utils/intelligent_cleaning.py` - Intelligent data cleaning
   - `utils/intelligent_plotting.py` - Intelligent plot generation

2. **Agents Refactored**:
   - `agents/analysis_agent.py` - Uses intelligent analysis
   - `agents/data_agent.py` - Uses intelligent cleaning
   - `agents/plot_agent.py` - Uses intelligent plotting

3. **Configuration**: No changes needed (backward compatible)

---

## Next Steps

### Recommended Enhancements

1. **Add Human-in-the-Loop**:
   - Approval gates before executing transformations
   - Interactive refinement of analysis plans
   - User feedback incorporation

2. **Persistent Memory**:
   - Learn from previous analyses
   - Improve prompts based on validation feedback
   - Build dataset-specific analysis templates

3. **Vision API for Plot Interpretation**:
   - Currently uses plot metadata only
   - Could use Claude vision to analyze plot images directly
   - More accurate visual interpretation

4. **Multi-Iteration Refinement**:
   - Currently single-pass validation
   - Could iterate 2-3 times for complex datasets
   - Balance cost vs. quality

5. **Performance Optimization**:
   - Cache dataset profiles
   - Reuse analysis plans for similar datasets
   - Parallel execution of independent analyses

---

## Conclusion

The multi-agent system now uses **Claude-directed intelligent analysis** instead of **hardcoded blind operations**. This fundamentally solves the "blindly creating mean" problem by ensuring analyses are:
- **Context-aware**: Appropriate for data semantics
- **Flexible**: Not limited to predefined operations
- **Safe**: Executed in sandboxed environment
- **Validated**: Checked for quality and reasonableness

All agents (Data, Plot, Analysis) now support flexible code execution with automatic fallback to basic mode for robustness.
