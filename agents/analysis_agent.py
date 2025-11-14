"""
PERSON 4: SHAMANTH - Analysis Agent
Handles: Statistical analysis, plot interpretation, insight generation
"""
import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from config.config import (
    CLEANED_DIR, PLOTS_DIR, SCHEMA_FILE,
    PLOT_METADATA_FILE, TRANSFORMATION_LOG_FILE,
    INSIGHTS_FILE, ANALYSIS_REPORT_FILE
)
from utils.claude_client import claude


class AnalysisAgent:
    """
    Analysis Agent
    Analyzes data and plots to generate insights and recommendations
    """

    def __init__(self):
        self.name = "analysis"

    def execute(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method called by CEO

        INPUT:
            - action: Action to perform (e.g., "generate_insights")
            - data: Dict with action parameters

        OUTPUT:
            - Dict with results (insights, report paths)
        """
        if action == "generate_insights":
            return self.generate_insights()
        else:
            raise ValueError(f"Unknown action: {action}")

    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights from data, schema, and plots

        INPUT:
            - None (reads from files)

        OUTPUT:
            - Dict {
                "insights_path": str,
                "report_path": str,
                "insights_count": int,
                "high_severity_count": int
              }
        """
        print(f"[AnalysisAgent] Generating insights...")

        # Load all inputs
        df, schema, plot_metadata, transform_log = self._load_inputs()

        # Perform statistical analysis
        stats = self._statistical_analysis(df, schema)

        # Interpret plots
        plot_insights = self._interpret_plots(df, schema, plot_metadata)

        # Generate data quality insights
        quality_insights = self._data_quality_insights(df, schema, transform_log)

        # Generate business insights
        business_insights = self._business_insights(df, schema, stats)

        # Combine all insights
        all_insights = quality_insights + plot_insights + business_insights

        # Generate recommendations
        recommendations = self._generate_recommendations(df, schema, all_insights)

        # Create structured insights JSON
        insights_data = self._create_insights_json(all_insights, recommendations)

        # Generate report
        report = self._generate_report(df, schema, insights_data, stats)

        # Save outputs
        output_paths = self._save_outputs(insights_data, report)

        print(f"[AnalysisAgent] Generated {len(all_insights)} insights")

        return {
            **output_paths,
            "insights_count": len(all_insights),
            "high_severity_count": sum(1 for i in all_insights if i.get("severity") == "high")
        }

    # ========================================
    # SHAMANTH: TODO - Implement these methods
    # ========================================

    def _load_inputs(self) -> tuple:
        """
        Load all required inputs

        INPUT:
            - None

        OUTPUT:
            - (df, schema, plot_metadata, transform_log)

        TODO: Load all files needed for analysis
        """
        print("[AnalysisAgent] TODO: Implement _load_inputs()")

        # PLACEHOLDER
        df = pd.read_csv(CLEANED_DIR / "cleaned_data.csv")

        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)

        with open(PLOT_METADATA_FILE, "r") as f:
            plot_metadata = json.load(f)

        with open(TRANSFORMATION_LOG_FILE, "r") as f:
            transform_log = json.load(f)

        return df, schema, plot_metadata, transform_log

    def _statistical_analysis(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform statistical analysis on data

        INPUT:
            - df: DataFrame
            - schema: Schema dict

        OUTPUT:
            - Dict with statistics (mean, median, std, skewness, anomalies, etc.)

        TODO: Use Claude to perform intelligent statistical analysis
        - Compute descriptive statistics
        - Detect skewness and kurtosis
        - Identify anomalies using statistical methods
        - Detect patterns and trends
        """
        print("[AnalysisAgent] Performing statistical analysis...")

        # Compute basic statistics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        stats = {
            "numeric_columns": {},
            "categorical_columns": {},
            "anomalies": [],
            "correlations": []
        }

        # Analyze numeric columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats["numeric_columns"][col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                    "null_count": int(df[col].isna().sum()),
                    "null_pct": float(df[col].isna().sum() / len(df) * 100)
                }

                # Detect outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    stats["anomalies"].append({
                        "column": col,
                        "type": "outliers",
                        "count": len(outliers),
                        "values": outliers.tolist()[:10]  # Limit to first 10
                    })

        # Analyze categorical columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            stats["categorical_columns"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": value_counts.head(10).to_dict(),
                "null_count": int(df[col].isna().sum()),
                "null_pct": float(df[col].isna().sum() / len(df) * 100)
            }

        # Compute correlations for numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            # Find strong correlations (|corr| > 0.7, excluding diagonal)
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        stats["correlations"].append({
                            "col1": numeric_cols[i],
                            "col2": numeric_cols[j],
                            "correlation": float(corr_val)
                        })

        # Use Claude to interpret the statistics
        prompt = f"""
Analyze these dataset statistics and provide intelligent insights:

Dataset Info:
- Total rows: {len(df)}
- Numeric columns: {len(numeric_cols)}
- Categorical columns: {len(categorical_cols)}

Numeric Column Statistics:
{json.dumps(stats["numeric_columns"], indent=2)}

Categorical Column Statistics:
{json.dumps(stats["categorical_columns"], indent=2)}

Detected Anomalies:
{json.dumps(stats["anomalies"], indent=2)}

Strong Correlations:
{json.dumps(stats["correlations"], indent=2)}

Please analyze these statistics and provide:
1. Key statistical patterns (distribution shapes, skewness interpretation)
2. Notable anomalies or outliers that need attention
3. Interesting correlations and what they might indicate
4. Data quality observations

Return your analysis as plain text, organized in clear sections.
"""

        try:
            claude_analysis = claude.call(prompt, max_tokens=2000)
            stats["claude_interpretation"] = claude_analysis
        except Exception as e:
            print(f"[AnalysisAgent] Warning: Claude analysis failed: {e}")
            stats["claude_interpretation"] = "Analysis not available"

        return stats

    def _interpret_plots(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        plot_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Interpret plots to generate insights

        INPUT:
            - df: DataFrame
            - schema: Schema dict
            - plot_metadata: Plot metadata with plot info

        OUTPUT:
            - List[Dict] with insights from plots

        TODO: Use Claude to interpret plots (vision API for images)
        - Analyze histogram shapes (normal, skewed, bimodal)
        - Identify outliers in plots
        - Detect patterns in time series
        - Interpret correlation heatmaps
        - Generate human-readable insights
        """
        print("[AnalysisAgent] Interpreting plots...")

        insights = []

        if "plots" not in plot_metadata or len(plot_metadata["plots"]) == 0:
            print("[AnalysisAgent] No plots found to interpret")
            return insights

        # Build a summary of all plots for Claude
        plot_summary = []
        for plot_info in plot_metadata["plots"]:
            plot_summary.append({
                "type": plot_info.get("type", "unknown"),
                "column": plot_info.get("column", plot_info.get("columns", [])),
                "description": plot_info.get("description", ""),
                "filename": plot_info.get("filename", "")
            })

        # Use Claude to interpret the plots based on metadata and data
        prompt = f"""
Analyze these data visualizations and generate insights:

Dataset Overview:
- Total rows: {len(df)}
- Columns: {df.columns.tolist()}

Available Plots:
{json.dumps(plot_summary, indent=2)}

Schema Information:
{json.dumps(schema.get("columns", [])[:10], indent=2)}

Based on the plot types and data, provide insights about:
1. Distribution patterns (for histograms): Are they normal, skewed, bimodal? What does this indicate?
2. Categorical patterns (for bar charts): Are there dominant categories? Any imbalances?
3. Correlation patterns (for heatmaps): Which variables are strongly related?
4. Time series patterns (if any): Are there trends, seasonality, or anomalies?
5. Outliers visible in the visualizations

For each insight, return a JSON array with this structure:
[
  {{
    "type": "trend|correlation|anomaly|pattern",
    "severity": "high|medium|low",
    "description": "Clear description of what the plot shows",
    "affected_columns": ["col1", "col2"],
    "recommendation": "Actionable recommendation based on the insight"
  }}
]

Return ONLY the JSON array, no other text.
"""

        try:
            response = claude.call(prompt, max_tokens=2000)
            # Try to parse the JSON response
            try:
                # Extract JSON from response (handle cases where Claude adds explanation)
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    parsed_insights = json.loads(json_match.group())
                    insights.extend(parsed_insights)
                else:
                    # Fallback: create a single insight from the text
                    insights.append({
                        "type": "pattern",
                        "severity": "medium",
                        "description": response[:500],
                        "affected_columns": [],
                        "recommendation": "Review the visualizations for detailed patterns"
                    })
            except json.JSONDecodeError:
                # If JSON parsing fails, create a general insight
                insights.append({
                    "type": "pattern",
                    "severity": "medium",
                    "description": f"Plot analysis: {response[:500]}",
                    "affected_columns": [],
                    "recommendation": "Review the visualizations for detailed patterns"
                })
        except Exception as e:
            print(f"[AnalysisAgent] Warning: Plot interpretation failed: {e}")

        return insights

    def _data_quality_insights(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        transform_log: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights about data quality

        INPUT:
            - df: DataFrame
            - schema: Schema dict
            - transform_log: Transformation log

        OUTPUT:
            - List[Dict] with data quality insights

        TODO: Analyze data quality
        - Missing data hotspots
        - Transformation impact
        - Data consistency issues
        - Encoding problems
        """
        print("[AnalysisAgent] Analyzing data quality...")

        insights = []

        # Analyze missing data
        missing_cols = []
        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 5:  # More than 5% missing
                severity = "high" if null_pct > 20 else "medium"
                missing_cols.append(col)
                insights.append({
                    "type": "quality",
                    "severity": severity,
                    "description": f"{null_pct:.1f}% missing values in '{col}' column",
                    "affected_columns": [col],
                    "recommendation": f"Review imputation strategy for '{col}'. High missingness may indicate data collection issues."
                })

        # Analyze transformation impact
        if "provenance" in transform_log:
            prov = transform_log["provenance"]
            original_rows = prov.get("original_rows", len(df))
            final_rows = prov.get("final_rows", len(df))
            rows_removed = original_rows - final_rows

            if rows_removed > 0:
                removal_pct = (rows_removed / original_rows) * 100
                severity = "high" if removal_pct > 10 else "medium" if removal_pct > 5 else "low"
                insights.append({
                    "type": "quality",
                    "severity": severity,
                    "description": f"Data cleaning removed {rows_removed} rows ({removal_pct:.1f}% of original data)",
                    "affected_columns": prov.get("columns_dropped", []),
                    "recommendation": "Review removed rows to ensure no important data was lost"
                })

        # Analyze high cardinality columns
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                unique_pct = (unique_count / len(df)) * 100
                if unique_pct > 50:  # More than 50% unique values
                    insights.append({
                        "type": "quality",
                        "severity": "low",
                        "description": f"High cardinality in '{col}': {unique_count} unique values ({unique_pct:.1f}%)",
                        "affected_columns": [col],
                        "recommendation": f"Consider if '{col}' should be treated as an identifier rather than a categorical variable"
                    })

        # Check schema warnings
        if "warnings" in schema and len(schema["warnings"]) > 0:
            for warning in schema["warnings"][:5]:  # Limit to first 5
                insights.append({
                    "type": "quality",
                    "severity": "medium",
                    "description": f"Schema warning: {warning}",
                    "affected_columns": [],
                    "recommendation": "Review data types and parsing logic"
                })

        # Analyze transformation operations
        if "operations" in transform_log:
            for op in transform_log["operations"]:
                rows_affected = op.get("rows_affected", 0)
                if rows_affected > 0:
                    affected_pct = (rows_affected / len(df)) * 100
                    if affected_pct > 10:  # More than 10% of data affected
                        insights.append({
                            "type": "quality",
                            "severity": "medium",
                            "description": f"Transformation '{op.get('operation')}' affected {rows_affected} rows ({affected_pct:.1f}%) in column '{op.get('column')}'",
                            "affected_columns": [op.get("column", "")],
                            "recommendation": f"Verify that '{op.get('operation')}' transformation was appropriate"
                        })

        return insights

    def _business_insights(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate business-level insights

        INPUT:
            - df: DataFrame
            - schema: Schema dict
            - stats: Statistical analysis results

        OUTPUT:
            - List[Dict] with business insights

        TODO: Use Claude to generate actionable business insights
        - Top anomalies worth investigating
        - Key trends and patterns
        - Interesting correlations
        - Revenue/cost drivers (if applicable)
        - Recommended actions
        """
        print("[AnalysisAgent] Generating business insights...")

        insights = []

        # Prepare data summary for Claude
        data_summary = {
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "numeric_stats": stats.get("numeric_columns", {}),
            "categorical_stats": stats.get("categorical_columns", {}),
            "anomalies": stats.get("anomalies", []),
            "correlations": stats.get("correlations", [])
        }

        # Get a sample of the data
        sample_data = df.head(5).to_dict('records')

        prompt = f"""
You are a data analyst generating business insights from a dataset. Analyze the following information:

Dataset Summary:
{json.dumps(data_summary, indent=2)}

Sample Records:
{json.dumps(sample_data, indent=2)}

Schema Information:
{json.dumps(schema.get("columns", [])[:10], indent=2)}

Generate 3-5 actionable business insights. Focus on:
1. Top anomalies worth investigating (unusual patterns, outliers)
2. Key trends and patterns that could inform business decisions
3. Interesting correlations between variables
4. Potential opportunities or risks identified in the data
5. Segments or categories that stand out

For each insight, return a JSON array with this structure:
[
  {{
    "type": "anomaly|trend|correlation|opportunity|risk",
    "severity": "high|medium|low",
    "description": "Clear, business-focused description of the insight",
    "affected_columns": ["col1", "col2"],
    "recommendation": "Specific, actionable recommendation"
  }}
]

Return ONLY the JSON array, no other text.
"""

        try:
            response = claude.call(prompt, max_tokens=2000)
            # Try to parse the JSON response
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    parsed_insights = json.loads(json_match.group())
                    insights.extend(parsed_insights)
                else:
                    # Fallback: create insights from text
                    insights.append({
                        "type": "trend",
                        "severity": "medium",
                        "description": response[:500],
                        "affected_columns": [],
                        "recommendation": "Review the data for these patterns"
                    })
            except json.JSONDecodeError:
                # Create a general insight
                insights.append({
                    "type": "trend",
                    "severity": "medium",
                    "description": f"Business analysis: {response[:500]}",
                    "affected_columns": [],
                    "recommendation": "Further investigation recommended"
                })
        except Exception as e:
            print(f"[AnalysisAgent] Warning: Business insights generation failed: {e}")

        # Add insights from correlations
        for corr in stats.get("correlations", [])[:3]:  # Top 3 correlations
            insights.append({
                "type": "correlation",
                "severity": "medium",
                "description": f"Strong correlation ({corr['correlation']:.2f}) between '{corr['col1']}' and '{corr['col2']}'",
                "affected_columns": [corr["col1"], corr["col2"]],
                "recommendation": f"Consider '{corr['col1']}' and '{corr['col2']}' relationship in modeling or business logic"
            })

        # Add insights from anomalies
        for anomaly in stats.get("anomalies", [])[:2]:  # Top 2 anomalies
            insights.append({
                "type": "anomaly",
                "severity": "high",
                "description": f"Detected {anomaly['count']} outliers in '{anomaly['column']}' column",
                "affected_columns": [anomaly["column"]],
                "recommendation": f"Investigate outliers in '{anomaly['column']}' - they may represent errors or important edge cases"
            })

        return insights

    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations for next steps

        INPUT:
            - df: DataFrame
            - schema: Schema dict
            - insights: All generated insights

        OUTPUT:
            - List[str] with recommendations

        TODO: Use Claude to generate smart recommendations
        - Suggest ML models to try
        - Identify key features for modeling
        - Propose additional data cleaning
        - Recommend data collection strategies
        """
        print("[AnalysisAgent] Generating recommendations...")

        # Prepare insights summary
        high_severity = [i for i in insights if i.get("severity") == "high"]
        medium_severity = [i for i in insights if i.get("severity") == "medium"]

        insights_summary = {
            "total": len(insights),
            "high_severity": len(high_severity),
            "medium_severity": len(medium_severity),
            "types": {}
        }

        for insight in insights:
            insight_type = insight.get("type", "unknown")
            insights_summary["types"][insight_type] = insights_summary["types"].get(insight_type, 0) + 1

        # Build prompt for Claude
        prompt = f"""
Based on the following data analysis, generate 5-7 actionable recommendations for next steps:

Dataset Information:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Column names: {df.columns.tolist()}

Schema Summary:
{json.dumps(schema.get("columns", [])[:10], indent=2)}

Insights Summary:
- Total insights: {insights_summary['total']}
- High severity: {insights_summary['high_severity']}
- Medium severity: {insights_summary['medium_severity']}
- Insight types: {insights_summary['types']}

High Priority Insights:
{json.dumps(high_severity[:5], indent=2)}

Generate recommendations for:
1. Machine learning models to try (based on data types and patterns)
2. Key features to focus on for modeling
3. Additional data cleaning or preprocessing steps
4. Data collection strategies to improve the dataset
5. Business actions based on the insights

Return a JSON array of strings:
[
  "Recommendation 1...",
  "Recommendation 2...",
  ...
]

Return ONLY the JSON array, no other text.
"""

        recommendations = []

        try:
            response = claude.call(prompt, max_tokens=1500)
            # Try to parse JSON
            try:
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    recommendations = json.loads(json_match.group())
                else:
                    # Split response into bullet points
                    lines = response.strip().split('\n')
                    recommendations = [line.strip('- ').strip() for line in lines if line.strip()][:7]
            except json.JSONDecodeError:
                # Split into lines
                lines = response.strip().split('\n')
                recommendations = [line.strip('- ').strip() for line in lines if line.strip()][:7]
        except Exception as e:
            print(f"[AnalysisAgent] Warning: Recommendation generation failed: {e}")
            recommendations = []

        # Add fallback recommendations based on insights
        if len(recommendations) == 0:
            if high_severity:
                recommendations.append(f"Address {len(high_severity)} high-severity issues identified in the analysis")

            # Check for specific patterns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 3:
                recommendations.append(f"Consider regression or clustering models with {len(numeric_cols)} numeric features")

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                recommendations.append(f"Apply encoding strategies for {len(categorical_cols)} categorical variables")

            recommendations.append("Review data quality issues before modeling")
            recommendations.append("Consider feature engineering based on domain knowledge")

        return recommendations[:7]  # Limit to 7 recommendations

    def _create_insights_json(
        self,
        insights: List[Dict[str, Any]],
        recommendations: List[str]
    ) -> Dict[str, Any]:
        """
        Create structured insights JSON

        INPUT:
            - insights: List of insight dicts
            - recommendations: List of recommendations

        OUTPUT:
            - Dict in insights.json format
        """
        return {
            "insights": insights,
            "summary": {
                "total_insights": len(insights),
                "high_severity": sum(1 for i in insights if i.get("severity") == "high"),
                "medium_severity": sum(1 for i in insights if i.get("severity") == "medium"),
                "low_severity": sum(1 for i in insights if i.get("severity") == "low"),
                "recommended_next_steps": recommendations
            }
        }

    def _generate_report(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        insights_data: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive analysis report in Markdown

        INPUT:
            - df: DataFrame
            - schema: Schema dict
            - insights_data: Insights JSON
            - stats: Statistical analysis

        OUTPUT:
            - str: Markdown report

        TODO: Use Claude to generate comprehensive, well-formatted report
        - Executive summary
        - Key findings
        - Visualizations referenced
        - Detailed insights
        - Recommendations
        """
        print("[AnalysisAgent] Generating comprehensive report...")

        # Prepare data for report
        insights = insights_data.get("insights", [])
        summary = insights_data.get("summary", {})

        # Group insights by type and severity
        high_severity = [i for i in insights if i.get("severity") == "high"]
        medium_severity = [i for i in insights if i.get("severity") == "medium"]
        low_severity = [i for i in insights if i.get("severity") == "low"]

        insights_by_type = {}
        for insight in insights:
            itype = insight.get("type", "other")
            if itype not in insights_by_type:
                insights_by_type[itype] = []
            insights_by_type[itype].append(insight)

        # Use Claude to generate the report
        prompt = f"""
Generate a comprehensive data analysis report in Markdown format based on the following information:

Dataset Overview:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Columns: {df.columns.tolist()}

Statistical Summary:
- Numeric columns: {len(stats.get('numeric_columns', {}))}
- Categorical columns: {len(stats.get('categorical_columns', {}))}
- Detected anomalies: {len(stats.get('anomalies', []))}
- Strong correlations: {len(stats.get('correlations', []))}

Claude's Statistical Interpretation:
{stats.get('claude_interpretation', 'Not available')[:1000]}

Insights Summary:
- Total insights: {summary.get('total_insights', 0)}
- High severity: {summary.get('high_severity', 0)}
- Medium severity: {summary.get('medium_severity', 0)}
- Low severity: {summary.get('low_severity', 0)}

High Priority Insights:
{json.dumps(high_severity[:5], indent=2)}

Medium Priority Insights:
{json.dumps(medium_severity[:5], indent=2)}

Recommendations:
{json.dumps(summary.get('recommended_next_steps', []), indent=2)}

Please create a well-structured markdown report with these sections:
1. Executive Summary (3-5 bullet points)
2. Dataset Overview (brief description of the data)
3. Key Findings (top 5-7 most important insights)
4. Statistical Analysis (summary of distributions, correlations, anomalies)
5. Data Quality Assessment (quality issues found)
6. Business Insights (actionable insights)
7. Visualizations Generated (list the types of plots created)
8. Recommendations (next steps)

Make the report professional, clear, and actionable. Use markdown formatting including headers, bullet points, and bold text for emphasis.
"""

        try:
            report_content = claude.call(prompt, max_tokens=3000)
        except Exception as e:
            print(f"[AnalysisAgent] Warning: Claude report generation failed: {e}")
            # Fallback to manual report generation
            report_content = self._generate_fallback_report(df, schema, insights_data, stats, insights_by_type)

        return report_content

    def _generate_fallback_report(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        insights_data: Dict[str, Any],
        stats: Dict[str, Any],
        insights_by_type: Dict[str, List[Dict]]
    ) -> str:
        """Generate a fallback report if Claude fails"""
        insights = insights_data.get("insights", [])
        summary = insights_data.get("summary", {})
        high_severity = [i for i in insights if i.get("severity") == "high"]

        report = f"""# Data Analysis Report

## Executive Summary

- Analyzed **{len(df)}** rows across **{len(df.columns)}** columns
- Generated **{summary.get('total_insights', 0)}** insights
- Identified **{summary.get('high_severity', 0)}** high-severity issues
- Found **{len(stats.get('anomalies', []))}** anomalies requiring attention
- Detected **{len(stats.get('correlations', []))}** strong correlations between variables

## Dataset Overview

The dataset contains {len(df.columns)} columns: {', '.join(df.columns.tolist()[:10])}{"..." if len(df.columns) > 10 else ""}.

**Column Types:**
- Numeric: {len(stats.get('numeric_columns', {}))} columns
- Categorical: {len(stats.get('categorical_columns', {}))} columns

## Key Findings

"""
        # Add top insights
        for i, insight in enumerate(high_severity[:5], 1):
            report += f"{i}. **[{insight.get('severity', '').upper()}]** {insight.get('description', '')}\n"
            report += f"   - *Recommendation:* {insight.get('recommendation', '')}\n\n"

        report += "## Statistical Analysis\n\n"

        # Add correlation info
        if stats.get('correlations'):
            report += "### Strong Correlations\n\n"
            for corr in stats.get('correlations', [])[:5]:
                report += f"- **{corr['col1']}** ↔ **{corr['col2']}**: {corr['correlation']:.2f}\n"
            report += "\n"

        # Add anomalies
        if stats.get('anomalies'):
            report += "### Detected Anomalies\n\n"
            for anomaly in stats.get('anomalies', [])[:5]:
                report += f"- **{anomaly['column']}**: {anomaly['count']} outliers detected\n"
            report += "\n"

        # Data quality
        report += "## Data Quality Assessment\n\n"
        quality_insights = insights_by_type.get('quality', [])
        if quality_insights:
            for insight in quality_insights[:5]:
                report += f"- {insight.get('description', '')}\n"
        else:
            report += "No major data quality issues detected.\n"
        report += "\n"

        # Business insights
        report += "## Business Insights\n\n"
        business_types = ['trend', 'correlation', 'anomaly', 'opportunity', 'risk']
        business_insights = [i for i in insights if i.get('type') in business_types]
        if business_insights:
            for insight in business_insights[:5]:
                report += f"- **{insight.get('type', '').title()}**: {insight.get('description', '')}\n"
        else:
            report += "Additional business context needed for deeper insights.\n"
        report += "\n"

        # Recommendations
        report += "## Recommendations\n\n"
        for i, rec in enumerate(summary.get('recommended_next_steps', [])[:7], 1):
            report += f"{i}. {rec}\n"

        report += "\n---\n\n*Report generated by Analysis Agent*\n"

        return report

    def _save_outputs(self, insights_data: Dict[str, Any], report: str) -> Dict[str, str]:
        """
        Save insights and report to files

        INPUT:
            - insights_data: Insights JSON
            - report: Report markdown

        OUTPUT:
            - Dict with paths to saved files
        """
        # Save insights JSON
        with open(INSIGHTS_FILE, "w") as f:
            json.dump(insights_data, f, indent=2)

        # Save report
        with open(ANALYSIS_REPORT_FILE, "w") as f:
            f.write(report)

        print(f"[AnalysisAgent] Saved outputs:")
        print(f"  - Insights: {INSIGHTS_FILE}")
        print(f"  - Report: {ANALYSIS_REPORT_FILE}")

        return {
            "insights_path": str(INSIGHTS_FILE),
            "report_path": str(ANALYSIS_REPORT_FILE)
        }
