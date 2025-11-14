"""
Intelligent Plotting System

Uses Claude to generate flexible plotting code that executes in sandbox.
"""

import json
from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
from utils.claude_client import ClaudeClient
from utils.sandbox import SandboxExecutor


class IntelligentPlotter:
    """
    Uses Claude to intelligently plan and execute plot generation.
    """

    def __init__(self, agent_name: str = "plot", plots_dir: Path = None):
        """
        Initialize intelligent plotter.

        Args:
            agent_name: Name of the agent for Claude client configuration
            plots_dir: Directory to save plots
        """
        self.claude = ClaudeClient(agent_name=agent_name)
        self.sandbox = SandboxExecutor(timeout=60)
        self.plots_dir = plots_dir

    def plan_plots(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Claude to plan appropriate visualizations.

        Args:
            df: DataFrame to visualize
            schema: Schema with column information

        Returns:
            Plot plan with recommended visualizations and code
        """
        # Sample data for context
        sample_data = df.head(10).to_dict(orient='records')

        prompt = f"""You are a data visualization expert. Based on the dataset characteristics, recommend appropriate visualizations and generate Python code to create them.

DATASET INFO:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}

SCHEMA INFORMATION:
{json.dumps(schema, indent=2, default=str)}

SAMPLE DATA (first 10 rows):
{json.dumps(sample_data, indent=2, default=str)}

INSTRUCTIONS:
1. Analyze the dataset characteristics and recommend visualizations that:
   - Are appropriate for the data types (don't make histograms of categorical data)
   - Reveal meaningful patterns and relationships
   - Follow data visualization best practices
   - Are diverse (not all the same type)

2. For each recommended plot, generate Python code that:
   - Uses matplotlib (plt), seaborn (sns), or plotly
   - Operates on DataFrame variable 'df'
   - Saves the plot to a file path in variable 'plot_path'
   - Creates plot metadata in variable 'plot_metadata'
   - Includes proper titles, labels, and formatting
   - Handles edge cases (empty data, missing values)

3. Return your response as a JSON object:
{{
  "recommended_plots": [
    {{
      "name": "distribution_of_age",
      "type": "histogram",
      "description": "Distribution of customer ages showing age demographics",
      "rationale": "Age is a continuous numeric variable suitable for histogram",
      "columns": ["age"],
      "code": "# Python code here using plt/sns\\nimport matplotlib.pyplot as plt\\nfig, ax = plt.subplots()\\n...",
      "filename": "age_distribution.png"
    }}
  ],
  "inappropriate_plots": [
    {{
      "name": "histogram_of_customer_id",
      "reason": "Customer ID is an identifier, not a meaningful numeric variable"
    }}
  ]
}}

AVAILABLE PLOT TYPES:
- histogram: For continuous numeric distributions
- box_plot: For numeric data with outliers
- scatter: For relationships between two numeric variables
- bar_chart: For categorical data or aggregated metrics
- line_chart: For time series or ordered data
- heatmap: For correlation matrices or pivot tables
- violin_plot: For distribution comparison across categories

IMPORTANT:
- Only recommend plots that make sense for the data
- Ensure code saves plots to files
- Include descriptive titles and labels
- Limit to 5-8 most informative plots

Return ONLY the JSON object, no additional text.
"""

        response = self.claude.call(prompt=prompt, max_tokens=4096)

        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean.split('```json')[1].split('```')[0].strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.split('```')[1].split('```')[0].strip()

            plan = json.loads(response_clean)
            return plan
        except json.JSONDecodeError as e:
            print(f"Failed to parse Claude's plot plan: {e}")
            return self._create_fallback_plan(df)

    def _create_fallback_plan(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic fallback plot plan"""
        plots = []

        # Find numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Histogram for first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            plots.append({
                'name': f'histogram_{col}',
                'type': 'histogram',
                'description': f'Distribution of {col}',
                'rationale': 'Fallback plot',
                'columns': [col],
                'code': f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
df['{col}'].dropna().hist(bins=30, ax=ax, edgecolor='black')
ax.set_title('Distribution of {col}')
ax.set_xlabel('{col}')
ax.set_ylabel('Frequency')
plt.tight_layout()

plot_path = '{self.plots_dir}/{col}_histogram.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)

plot_metadata = {{
    'type': 'histogram',
    'column': '{col}',
    'filename': '{col}_histogram.png'
}}
""",
                'filename': f'{col}_histogram.png'
            })

        # Bar chart for first categorical column
        if categorical_cols:
            col = categorical_cols[0]
            plots.append({
                'name': f'bar_{col}',
                'type': 'bar',
                'description': f'Frequency of {col} categories',
                'rationale': 'Fallback plot',
                'columns': [col],
                'code': f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
value_counts = df['{col}'].value_counts().head(10)
value_counts.plot(kind='bar', ax=ax, edgecolor='black')
ax.set_title('Top Categories in {col}')
ax.set_xlabel('{col}')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plot_path = '{self.plots_dir}/{col}_bar.png'
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close(fig)

plot_metadata = {{
    'type': 'bar',
    'column': '{col}',
    'filename': '{col}_bar.png'
}}
""",
                'filename': f'{col}_bar.png'
            })

        return {
            'recommended_plots': plots,
            'inappropriate_plots': [],
            'warnings': ['Using fallback plot plan']
        }

    def execute_plots(
        self,
        df: pd.DataFrame,
        plot_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute plot generation in sandbox.

        Args:
            df: DataFrame to visualize
            plot_plan: Plan from plan_plots()

        Returns:
            Dictionary with plot metadata and execution results
        """
        results = {
            'successful_plots': [],
            'failed_plots': [],
            'plot_metadata': []
        }

        for plot_info in plot_plan.get('recommended_plots', []):
            code = plot_info['code']
            name = plot_info['name']

            print(f"[IntelligentPlotter] Generating plot: {name}")

            # Execute in sandbox
            success, result, stdout, stderr = self.sandbox.execute(
                code=code,
                data_context={'df': df},
                return_variable='plot_metadata'
            )

            if success and result:
                results['successful_plots'].append({
                    'name': name,
                    'description': plot_info['description'],
                    'metadata': result
                })
                results['plot_metadata'].append(result)
            else:
                results['failed_plots'].append({
                    'name': name,
                    'description': plot_info['description'],
                    'error': stderr
                })
                print(f"[IntelligentPlotter]   WARNING: Plot generation failed: {stderr[:200]}")

        print(f"[IntelligentPlotter] Generated {len(results['successful_plots'])}/{len(plot_plan.get('recommended_plots', []))} plots successfully")

        return results
