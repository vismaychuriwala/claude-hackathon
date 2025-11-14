"""
Test script for Analysis Agent
Creates mock data and tests the agent functionality
"""
import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.analysis_agent import AnalysisAgent
from config.config import (
    CLEANED_DIR, SCHEMA_FILE, PLOT_METADATA_FILE,
    TRANSFORMATION_LOG_FILE, OUTPUT_DIR
)


def setup_mock_data():
    """Create mock data files for testing"""
    print("Setting up mock data...")

    # Create output directories
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Create cleaned_data.csv
    mock_data = pd.DataFrame({
        'employee_id': range(1, 51),
        'name': [f'Employee_{i}' for i in range(1, 51)],
        'age': [25, 30, 35, 28, 32, 45, 29, 31, 27, 33] * 5,
        'salary': [50000, 60000, 70000, 55000, 65000, 80000, 52000, 68000, 51000, 72000] * 5,
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'HR'] * 10,
        'years_experience': [2, 5, 8, 3, 6, 15, 4, 7, 2, 9] * 5,
        'performance_score': [85, 90, 88, 92, 78, 95, 82, 89, 87, 91] * 5
    })

    # Add some outliers
    mock_data.loc[0, 'salary'] = 150000  # Outlier
    mock_data.loc[5, 'salary'] = 200000  # Outlier

    cleaned_file = CLEANED_DIR / "cleaned_data.csv"
    mock_data.to_csv(cleaned_file, index=False)
    print(f"✓ Created {cleaned_file}")

    # 2. Create schema.json
    schema = {
        "columns": [
            {
                "name": "employee_id",
                "type": "numeric",
                "null_pct": 0,
                "unique_count": 50
            },
            {
                "name": "name",
                "type": "text",
                "null_pct": 0,
                "unique_count": 50
            },
            {
                "name": "age",
                "type": "numeric",
                "null_pct": 0,
                "min": 25,
                "max": 45
            },
            {
                "name": "salary",
                "type": "numeric",
                "null_pct": 0,
                "min": 50000,
                "max": 200000
            },
            {
                "name": "department",
                "type": "categorical",
                "null_pct": 0,
                "unique_count": 5
            },
            {
                "name": "years_experience",
                "type": "numeric",
                "null_pct": 0,
                "min": 2,
                "max": 15
            },
            {
                "name": "performance_score",
                "type": "numeric",
                "null_pct": 0,
                "min": 78,
                "max": 95
            }
        ],
        "warnings": [
            "Detected 2 salary outliers above 99th percentile"
        ]
    }

    with open(SCHEMA_FILE, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"✓ Created {SCHEMA_FILE}")

    # 3. Create plot_metadata.json
    plot_metadata = {
        "plots": [
            {
                "filename": "histogram_salary.png",
                "type": "histogram",
                "column": "salary",
                "description": "Distribution of salaries shows right skew with two outliers"
            },
            {
                "filename": "histogram_age.png",
                "type": "histogram",
                "column": "age",
                "description": "Age distribution appears roughly uniform"
            },
            {
                "filename": "bar_department.png",
                "type": "bar",
                "column": "department",
                "description": "Employees distributed evenly across departments"
            },
            {
                "filename": "correlation_heatmap.png",
                "type": "heatmap",
                "columns": ["age", "salary", "years_experience", "performance_score"],
                "description": "Correlation matrix of numeric variables"
            },
            {
                "filename": "scatter_experience_salary.png",
                "type": "scatter",
                "columns": ["years_experience", "salary"],
                "description": "Relationship between experience and salary"
            }
        ],
        "total_plots": 5
    }

    with open(PLOT_METADATA_FILE, 'w') as f:
        json.dump(plot_metadata, f, indent=2)
    print(f"✓ Created {PLOT_METADATA_FILE}")

    # 4. Create transformation_log.json
    transform_log = {
        "operations": [
            {
                "step": 1,
                "operation": "cast_column",
                "column": "employee_id",
                "from_type": "object",
                "to_type": "int",
                "rows_affected": 0,
                "errors": 0
            },
            {
                "step": 2,
                "operation": "remove_duplicates",
                "columns": ["employee_id"],
                "rows_affected": 2,
                "rows_removed": 2
            },
            {
                "step": 3,
                "operation": "impute_missing",
                "column": "performance_score",
                "method": "median",
                "rows_affected": 3
            }
        ],
        "provenance": {
            "original_rows": 52,
            "final_rows": 50,
            "columns_dropped": [],
            "timestamp": "2025-01-14T15:30:00Z"
        }
    }

    with open(TRANSFORMATION_LOG_FILE, 'w') as f:
        json.dump(transform_log, f, indent=2)
    print(f"✓ Created {TRANSFORMATION_LOG_FILE}")

    print("\n" + "="*50)
    print("Mock data setup complete!")
    print("="*50 + "\n")


def test_analysis_agent():
    """Test the Analysis Agent"""
    print("Testing Analysis Agent...\n")

    # Create agent instance
    agent = AnalysisAgent()

    # Execute the agent
    try:
        result = agent.execute("generate_insights", {})

        print("\n" + "="*50)
        print("✓ Analysis Agent execution completed!")
        print("="*50)

        print("\nResults:")
        print(f"  - Insights generated: {result.get('insights_count', 0)}")
        print(f"  - High severity insights: {result.get('high_severity_count', 0)}")
        print(f"  - Insights file: {result.get('insights_path', 'N/A')}")
        print(f"  - Report file: {result.get('report_path', 'N/A')}")

        # Verify outputs exist
        print("\nVerifying outputs...")
        from config.config import INSIGHTS_FILE, ANALYSIS_REPORT_FILE

        if INSIGHTS_FILE.exists():
            print(f"  ✓ {INSIGHTS_FILE} exists")
            with open(INSIGHTS_FILE, 'r') as f:
                insights = json.load(f)
                print(f"    - Total insights: {insights['summary']['total_insights']}")
                print(f"    - High severity: {insights['summary']['high_severity']}")
                print(f"    - Recommendations: {len(insights['summary']['recommended_next_steps'])}")
        else:
            print(f"  ✗ {INSIGHTS_FILE} not found")

        if ANALYSIS_REPORT_FILE.exists():
            print(f"  ✓ {ANALYSIS_REPORT_FILE} exists")
            with open(ANALYSIS_REPORT_FILE, 'r') as f:
                report = f.read()
                print(f"    - Report length: {len(report)} characters")
                print(f"    - Report preview (first 200 chars):")
                print(f"      {report[:200]}...")
        else:
            print(f"  ✗ {ANALYSIS_REPORT_FILE} not found")

        print("\n" + "="*50)
        print("✓ Test completed successfully!")
        print("="*50)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*50)
    print("Analysis Agent Test Script")
    print("="*50 + "\n")

    # Setup mock data
    setup_mock_data()

    # Test the agent
    success = test_analysis_agent()

    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed. Check the error messages above.")
