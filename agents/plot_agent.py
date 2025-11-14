"""
PERSON 3: NIKUNJ - Visualization Agent
Handles: Automatic plot generation from cleaned data
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from config.config import CLEANED_DIR, PLOTS_DIR, SCHEMA_FILE, PLOT_METADATA_FILE
from utils.claude_client import claude


class PlotAgent:
    """
    Visualization Agent
    Creates automatic plots based on data types and schema
    """

    def __init__(self):
        self.name = "plot"

    def execute(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method called by CEO

        INPUT:
            - action: Action to perform (e.g., "create_plots")
            - data: Dict with action parameters

        OUTPUT:
            - Dict with results (plot paths, metadata)
        """
        if action == "create_plots":
            return self.create_plots(data.get("cleaned_data_path"))
        else:
            raise ValueError(f"Unknown action: {action}")

    def create_plots(self, cleaned_data_path: str = None) -> Dict[str, Any]:
        """
        Generate all plots from cleaned data

        INPUT:
            - cleaned_data_path: Path to cleaned CSV (optional, uses default if None)

        OUTPUT:
            - Dict {
                "plot_metadata_path": str,
                "plots": [{"filename": str, "type": str, "description": str}],
                "total_plots": int
              }
        """
        print(f"[PlotAgent] Creating plots...")

        # Load data and schema
        df, schema = self._load_data_and_schema(cleaned_data_path)

        # Determine what plots to create
        plot_plan = self._plan_plots(df, schema)

        # Generate plots
        plots_metadata = []
        for plan_item in plot_plan:
            plot_info = self._generate_plot(df, plan_item)
            plots_metadata.append(plot_info)

        # Save metadata
        metadata_path = self._save_metadata(plots_metadata)

        print(f"[PlotAgent] Created {len(plots_metadata)} plots")

        return {
            "plot_metadata_path": str(metadata_path),
            "plots": plots_metadata,
            "total_plots": len(plots_metadata)
        }

    # ========================================
    # NIKUNJ: TODO - Implement these methods
    # ========================================

    def _load_data_and_schema(self, cleaned_data_path: str = None) -> tuple:
        """
        Load cleaned data and schema

        INPUT:
            - cleaned_data_path: Optional path to cleaned CSV

        OUTPUT:
            - (df, schema): DataFrame and schema dict

        TODO: Load data and schema from files
        """
        print("[PlotAgent] TODO: Implement _load_data_and_schema()")

        # PLACEHOLDER
        if cleaned_data_path is None:
            cleaned_data_path = CLEANED_DIR / "cleaned_data.csv"

        df = pd.read_csv(cleaned_data_path)

        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)

        return df, schema

    def _plan_plots(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decide what plots to create based on schema

        INPUT:
            - df: DataFrame
            - schema: Schema dict with column types

        OUTPUT:
            - List[Dict] with plot plans: [
                {"type": "histogram", "column": "age", "title": "..."},
                {"type": "bar", "column": "category", "title": "..."},
                ...
              ]

        TODO: Use Claude to intelligently plan plots
        - Histogram for numeric columns
        - Bar charts for categorical columns
        - Time series for datetime columns
        - Correlation heatmap for numeric columns
        - Missingness heatmap
        """
        print("[PlotAgent] TODO: Implement _plan_plots() using Claude")

        # PLACEHOLDER: Create basic plan
        plot_plan = []

        # Example: Create histogram for first numeric column
        for col_info in schema.get("columns", []):
            if "int" in col_info["type"].lower() or "float" in col_info["type"].lower():
                plot_plan.append({
                    "type": "histogram",
                    "column": col_info["name"],
                    "title": f"Distribution of {col_info['name']}"
                })
                break  # Just one for now

        return plot_plan

    def _generate_plot(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single plot based on plan

        INPUT:
            - df: DataFrame
            - plan: Dict with plot spec {"type": str, "column": str, ...}

        OUTPUT:
            - Dict {
                "filename": str,
                "type": str,
                "column": str,
                "description": str,
                "insights": [str]
              }

        TODO: Implement plot generation
        - Create plot using matplotlib/plotly
        - Save as PNG and HTML (interactive)
        - Use Claude to generate description and insights
        """
        print(f"[PlotAgent] TODO: Implement _generate_plot() for {plan['type']}")

        # PLACEHOLDER: Create simple plot
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_type = plan["type"]
        column = plan["column"]
        filename = f"{plot_type}_{column}.png"
        filepath = PLOTS_DIR / filename

        # Create basic plot
        plt.figure(figsize=(10, 6))

        if plot_type == "histogram":
            plt.hist(df[column].dropna(), bins=30, edgecolor='black')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(plan.get("title", f"Histogram of {column}"))

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()

        return {
            "filename": filename,
            "type": plot_type,
            "column": column,
            "description": f"TODO: Use Claude to generate description for {plot_type} of {column}",
            "insights": ["TODO: Use Claude to extract insights from plot"]
        }

    def _save_metadata(self, plots_metadata: List[Dict[str, Any]]) -> Path:
        """
        Save plot metadata to JSON

        INPUT:
            - plots_metadata: List of plot info dicts

        OUTPUT:
            - Path to saved metadata file
        """
        metadata = {
            "plots": plots_metadata,
            "total_plots": len(plots_metadata)
        }

        with open(PLOT_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[PlotAgent] Saved metadata: {PLOT_METADATA_FILE}")
        return PLOT_METADATA_FILE
