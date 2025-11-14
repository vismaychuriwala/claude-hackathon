"""
PERSON 3: NIKUNJ - Visualization Agent
Handles: Automatic plot generation from cleaned data
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List
from config.config import CLEANED_DIR, PLOTS_DIR, SCHEMA_FILE, PLOT_METADATA_FILE
from utils.claude_client import claude
from utils.intelligent_plotting import IntelligentPlotter


class PlotAgent:
    """
    Visualization Agent
    Creates automatic plots based on data types and schema
    """

    def __init__(self):
        self.name = "plot"
        self.intelligent_plotter = IntelligentPlotter(agent_name="plot", plots_dir=PLOTS_DIR)

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
        Generate all plots using INTELLIGENT plotting system.

        INPUT:
            - cleaned_data_path: Path to cleaned CSV (optional, uses default if None)

        OUTPUT:
            - Dict {
                "plot_metadata_path": str,
                "plots": [{"filename": str, "type": str, "description": str}],
                "total_plots": int
              }

        IMPROVEMENT OVER OLD VERSION:
        - Claude generates flexible plotting code instead of using fixed plot templates
        - Can create custom visualizations beyond predefined types
        - Intelligent selection based on data characteristics
        """
        print(f"[PlotAgent] Creating plots with intelligent plotter...")

        try:
            # Load data and schema
            df, schema = self._load_data_and_schema(cleaned_data_path)

            # Step 1: Plan plots with Claude
            print("[PlotAgent]   Step 1/2: Planning visualizations...")
            plot_plan = self.intelligent_plotter.plan_plots(df=df, schema=schema)

            print(f"[PlotAgent]   Planned {len(plot_plan.get('recommended_plots', []))} plots")
            if plot_plan.get('inappropriate_plots'):
                print(f"[PlotAgent]   Avoided {len(plot_plan['inappropriate_plots'])} inappropriate plots")

            # Step 2: Execute plot generation
            print("[PlotAgent]   Step 2/2: Generating plots...")
            execution_results = self.intelligent_plotter.execute_plots(
                df=df,
                plot_plan=plot_plan
            )

            # Extract plot metadata
            plots_metadata = execution_results.get('plot_metadata', [])

            # Save metadata
            metadata_path = self._save_metadata(plots_metadata)

            print(f"[PlotAgent] ✓ Created {len(plots_metadata)} plots successfully")

            return {
                "plot_metadata_path": str(metadata_path),
                "plots": plots_metadata,
                "total_plots": len(plots_metadata),
                "intelligent_mode": True
            }

        except Exception as e:
            print(f"[PlotAgent] ERROR: Intelligent plotting failed: {e}")
            print("[PlotAgent] Falling back to basic plotting...")
            return self._fallback_create_plots(cleaned_data_path)

    def _fallback_create_plots(self, cleaned_data_path: str = None) -> Dict[str, Any]:
        """Fallback to basic plotting if intelligent mode fails"""
        df, schema = self._load_data_and_schema(cleaned_data_path)

        # Create basic plots using old method
        plot_plan = self._plan_plots(df, schema)

        plots_metadata = []
        for plan_item in plot_plan:
            try:
                plot_info = self._generate_plot(df, plan_item)
                plots_metadata.append(plot_info)
            except Exception as e:
                print(f"[PlotAgent] Failed to generate plot: {e}")

        metadata_path = self._save_metadata(plots_metadata)

        return {
            "plot_metadata_path": str(metadata_path),
            "plots": plots_metadata,
            "total_plots": len(plots_metadata),
            "intelligent_mode": False,
            "fallback_reason": "Intelligent plotting failed"
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

        Loads data with robust error handling
        """
        print("[PlotAgent] Loading data and schema...")

        # Determine data path
        if cleaned_data_path is None:
            cleaned_data_path = CLEANED_DIR / "cleaned_data.csv"
        else:
            cleaned_data_path = Path(cleaned_data_path)

        # Validate file exists
        if not cleaned_data_path.exists():
            raise FileNotFoundError(f"Cleaned data not found at: {cleaned_data_path}")

        # Load CSV with error handling
        try:
            df = pd.read_csv(cleaned_data_path)
            print(f"[PlotAgent] Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Failed to load CSV from {cleaned_data_path}: {str(e)}")

        # Validate data is not empty
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        # Load schema
        if not SCHEMA_FILE.exists():
            raise FileNotFoundError(f"Schema file not found at: {SCHEMA_FILE}")

        try:
            with open(SCHEMA_FILE, "r") as f:
                schema = json.load(f)
            print(f"[PlotAgent] Loaded schema with {len(schema.get('columns', []))} columns")
        except Exception as e:
            raise ValueError(f"Failed to load schema from {SCHEMA_FILE}: {str(e)}")

        # Validate schema has required structure
        if "columns" not in schema:
            raise ValueError("Schema missing 'columns' key")

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

        Uses Claude to intelligently plan plots based on data characteristics
        """
        print("[PlotAgent] Planning plots using Claude...")

        # Prepare data summary for Claude
        data_summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": []
        }

        for col_info in schema.get("columns", []):
            col_name = col_info["name"]
            col_type = col_info["type"]

            col_summary = {
                "name": col_name,
                "type": col_type,
                "null_count": int(df[col_name].isnull().sum()),
                "null_percentage": round(float(df[col_name].isnull().sum() / len(df) * 100), 2)
            }

            # Add type-specific stats
            if "int" in col_type.lower() or "float" in col_type.lower():
                col_summary["unique_values"] = int(df[col_name].nunique())
                col_summary["min"] = float(df[col_name].min()) if not df[col_name].isnull().all() else None
                col_summary["max"] = float(df[col_name].max()) if not df[col_name].isnull().all() else None
                col_summary["mean"] = float(df[col_name].mean()) if not df[col_name].isnull().all() else None
            elif "object" in col_type.lower() or "string" in col_type.lower():
                col_summary["unique_values"] = int(df[col_name].nunique())
                col_summary["top_values"] = df[col_name].value_counts().head(5).to_dict()
            elif "datetime" in col_type.lower():
                col_summary["unique_values"] = int(df[col_name].nunique())
                if not df[col_name].isnull().all():
                    col_summary["min"] = str(df[col_name].min())
                    col_summary["max"] = str(df[col_name].max())

            data_summary["columns"].append(col_summary)

        # Create prompt for Claude
        prompt = f"""You are a data visualization expert. Based on the following dataset summary, suggest 3-4 insightful visualizations.

Dataset Summary:
{json.dumps(data_summary, indent=2)}

Please suggest visualizations that would be most insightful for understanding this data.

Available plot types:
- histogram: For numeric distributions
- bar: For categorical data or counts
- scatter: For relationships between two numeric variables
- correlation: For correlation matrix of numeric variables
- time_series: For temporal trends
- box: For numeric distribution with outliers

Return ONLY a JSON array with 3-4 plot suggestions in this exact format:
[
  {{
    "type": "histogram",
    "column": "column_name",
    "title": "Descriptive title",
    "x_column": "column_name",
    "y_column": "column_name"
  }}
]

Notes:
- For scatter plots, include both x_column and y_column
- For correlation plots, set column to "all_numeric"
- For single-variable plots, use column field
- Prioritize the most insightful visualizations
- Limit to 3-4 plots maximum
"""

        # Call Claude
        try:
            response = claude.call(prompt, max_tokens=2048)

            # Parse response - extract JSON from response
            response_text = response.strip()

            # Try to find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                plot_plan = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in Claude response")

            print(f"[PlotAgent] Claude suggested {len(plot_plan)} plots")
            return plot_plan

        except Exception as e:
            print(f"[PlotAgent] Warning: Claude planning failed ({str(e)}), using fallback strategy")
            return self._fallback_plot_plan(df, schema)

    def _fallback_plot_plan(self, df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback plot planning if Claude fails"""
        plot_plan = []

        # Find numeric and categorical columns
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []

        for col_info in schema.get("columns", []):
            col_name = col_info["name"]
            col_type = col_info["type"].lower()

            if "int" in col_type or "float" in col_type:
                numeric_cols.append(col_name)
            elif "datetime" in col_type:
                datetime_cols.append(col_name)
            elif "object" in col_type or "string" in col_type:
                if df[col_name].nunique() < 20:  # Only if not too many categories
                    categorical_cols.append(col_name)

        # Create up to 4 plots
        if numeric_cols:
            plot_plan.append({
                "type": "histogram",
                "column": numeric_cols[0],
                "title": f"Distribution of {numeric_cols[0]}"
            })

        if categorical_cols:
            plot_plan.append({
                "type": "bar",
                "column": categorical_cols[0],
                "title": f"Count by {categorical_cols[0]}"
            })

        if len(numeric_cols) >= 2:
            plot_plan.append({
                "type": "scatter",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1],
                "title": f"{numeric_cols[0]} vs {numeric_cols[1]}"
            })

        if len(numeric_cols) >= 3:
            plot_plan.append({
                "type": "correlation",
                "column": "all_numeric",
                "title": "Correlation Matrix"
            })

        return plot_plan[:4]  # Limit to 4 plots

    def _generate_plot(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single plot based on plan

        INPUT:
            - df: DataFrame
            - plan: Dict with plot spec {"type": str, "column": str, ...}

        OUTPUT:
            - Dict {
                "filename": str,
                "filename_html": str,
                "type": str,
                "column": str,
                "description": str,
                "insights": [str]
              }

        Creates both static (PNG) and interactive (HTML) versions
        """
        print(f"[PlotAgent] Generating {plan['type']} plot...")

        # Ensure plots directory exists
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        plot_type = plan["type"]
        column = plan.get("column", "")
        title = plan.get("title", f"{plot_type.title()} Plot")

        # Generate unique filename
        if plot_type == "scatter":
            base_name = f"{plot_type}_{plan.get('x_column', '')}_{plan.get('y_column', '')}"
        elif plot_type == "correlation":
            base_name = "correlation_matrix"
        else:
            base_name = f"{plot_type}_{column}"

        filename_png = f"{base_name}.png"
        filename_html = f"{base_name}.html"
        filepath_png = PLOTS_DIR / filename_png
        filepath_html = PLOTS_DIR / filename_html

        # Generate plot based on type
        try:
            if plot_type == "histogram":
                self._create_histogram(df, plan, filepath_png, filepath_html)
            elif plot_type == "bar":
                self._create_bar_chart(df, plan, filepath_png, filepath_html)
            elif plot_type == "scatter":
                self._create_scatter_plot(df, plan, filepath_png, filepath_html)
            elif plot_type == "correlation":
                self._create_correlation_plot(df, plan, filepath_png, filepath_html)
            elif plot_type == "time_series":
                self._create_time_series(df, plan, filepath_png, filepath_html)
            elif plot_type == "box":
                self._create_box_plot(df, plan, filepath_png, filepath_html)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            print(f"[PlotAgent] Saved: {filename_png} and {filename_html}")

        except Exception as e:
            print(f"[PlotAgent] Error generating plot: {str(e)}")
            raise

        # Generate description and insights using Claude
        description, insights = self._generate_insights(df, plan)

        return {
            "filename": filename_png,
            "filename_html": filename_html,
            "type": plot_type,
            "column": column,
            "description": description,
            "insights": insights
        }

    def _create_histogram(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create histogram in both matplotlib and plotly"""
        column = plan["column"]
        title = plan.get("title", f"Distribution of {column}")

        # Matplotlib version
        plt.figure(figsize=(10, 6))
        data = df[column].dropna()
        plt.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = px.histogram(df, x=column, title=title, nbins=30)
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency",
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _create_bar_chart(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create bar chart in both matplotlib and plotly"""
        column = plan["column"]
        title = plan.get("title", f"Count by {column}")

        # Get value counts
        value_counts = df[column].value_counts().head(20)  # Limit to top 20

        # Matplotlib version
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = px.bar(x=value_counts.index, y=value_counts.values, title=title)
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _create_scatter_plot(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create scatter plot in both matplotlib and plotly"""
        x_col = plan.get("x_column")
        y_col = plan.get("y_column")
        title = plan.get("title", f"{x_col} vs {y_col}")

        # Matplotlib version
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_col], df[y_col], alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = px.scatter(df, x=x_col, y=y_col, title=title, opacity=0.6)
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _create_correlation_plot(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create correlation heatmap in both matplotlib and plotly"""
        title = plan.get("title", "Correlation Matrix")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation plot")

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Matplotlib version with seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title=title,
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _create_time_series(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create time series plot in both matplotlib and plotly"""
        column = plan["column"]
        title = plan.get("title", f"Time Series of {column}")

        # Assume index is datetime or find a datetime column
        time_col = plan.get("time_column")
        if time_col:
            df_plot = df[[time_col, column]].dropna()
            x_data = pd.to_datetime(df_plot[time_col])
            y_data = df_plot[column]
        else:
            # Use index if it's datetime or first datetime column
            df_plot = df[[column]].dropna()
            x_data = df_plot.index
            y_data = df_plot[column]

        # Matplotlib version
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, y_data, color='steelblue', linewidth=2)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(column, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = px.line(x=x_data, y=y_data, title=title)
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=column,
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _create_box_plot(self, df: pd.DataFrame, plan: Dict[str, Any], png_path: Path, html_path: Path):
        """Create box plot in both matplotlib and plotly"""
        column = plan["column"]
        title = plan.get("title", f"Box Plot of {column}")

        # Matplotlib version
        plt.figure(figsize=(8, 6))
        plt.boxplot(df[column].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        plt.ylabel(column, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Plotly version
        fig = px.box(df, y=column, title=title)
        fig.update_layout(
            template="plotly_white",
            font=dict(size=12)
        )
        fig.write_html(html_path)

    def _generate_insights(self, df: pd.DataFrame, plan: Dict[str, Any]) -> tuple:
        """Use Claude to generate description and insights for a plot"""
        print("[PlotAgent] Generating insights with Claude...")

        plot_type = plan["type"]
        column = plan.get("column", "")

        # Prepare data summary for Claude
        if plot_type == "scatter":
            x_col = plan.get("x_column")
            y_col = plan.get("y_column")
            data_stats = {
                "plot_type": plot_type,
                "x_column": x_col,
                "y_column": y_col,
                "x_stats": {
                    "mean": float(df[x_col].mean()),
                    "std": float(df[x_col].std()),
                    "min": float(df[x_col].min()),
                    "max": float(df[x_col].max())
                },
                "y_stats": {
                    "mean": float(df[y_col].mean()),
                    "std": float(df[y_col].std()),
                    "min": float(df[y_col].min()),
                    "max": float(df[y_col].max())
                },
                "correlation": float(df[[x_col, y_col]].corr().iloc[0, 1])
            }
        elif plot_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": float(corr_matrix.iloc[i, j])
                    })
            corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            data_stats = {
                "plot_type": plot_type,
                "top_correlations": corr_pairs[:5]
            }
        elif plot_type == "bar":
            value_counts = df[column].value_counts().head(10)
            data_stats = {
                "plot_type": plot_type,
                "column": column,
                "total_unique": int(df[column].nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()}
            }
        else:
            # For histogram, box, time_series
            data_stats = {
                "plot_type": plot_type,
                "column": column,
                "mean": float(df[column].mean()) if df[column].dtype in ['int64', 'float64'] else None,
                "std": float(df[column].std()) if df[column].dtype in ['int64', 'float64'] else None,
                "min": float(df[column].min()) if df[column].dtype in ['int64', 'float64'] else None,
                "max": float(df[column].max()) if df[column].dtype in ['int64', 'float64'] else None,
                "null_count": int(df[column].isnull().sum())
            }

        prompt = f"""You are a data analyst. Generate insights for a visualization.

Plot Information:
{json.dumps(data_stats, indent=2)}

Provide:
1. A concise 1-2 sentence description of what this plot shows
2. 2-3 key insights or observations from the data

Return your response in this exact JSON format:
{{
  "description": "Brief description of the plot",
  "insights": [
    "First insight",
    "Second insight",
    "Third insight"
  ]
}}
"""

        try:
            response = claude.call(prompt, max_tokens=2048)

            # Parse JSON from response
            response_text = response.strip()
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return result.get("description", ""), result.get("insights", [])
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            print(f"[PlotAgent] Warning: Insight generation failed ({str(e)}), using fallback")
            # Fallback
            description = f"{plot_type.title()} plot of {column if column else 'data'}"
            insights = [f"Visualizing {plot_type} for the dataset"]
            return description, insights

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
