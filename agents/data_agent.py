"""
PERSON 2: VISMAY - Data Pipeline Agent
Handles: Ingestion, Preprocessing, Schema Inference, Data Cleaning
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from config.config import (
    RAW_DIR, CLEANED_DIR, SCHEMA_FILE,
    TRANSFORMATION_LOG_FILE, DATA_QUALITY_REPORT_FILE
)
from utils.claude_client import claude


class DataAgent:
    """
    Data Pipeline Agent
    Handles file ingestion, preprocessing, schema inference, and cleaning
    """

    def __init__(self):
        self.name = "data"

    def execute(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method called by CEO

        INPUT:
            - action: Action to perform (e.g., "process_file")
            - data: Dict with action parameters

        OUTPUT:
            - Dict with results (file paths, stats, etc.)
        """
        if action == "process_file":
            return self.process_file(data["file_path"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Full pipeline: Ingest -> Preprocess -> Schema -> Clean

        INPUT:
            - file_path: Path to uploaded file

        OUTPUT:
            - Dict {
                "cleaned_data_path": str,
                "schema_path": str,
                "transformation_log_path": str,
                "report_path": str,
                "stats": {...}
              }
        """
        print(f"[DataAgent] Processing file: {file_path}")

        # Step 1: Ingest
        file_info = self._ingest_file(file_path)

        # Step 2: Preprocess
        df = self._preprocess(file_path, file_info)

        # Step 3: Infer schema
        schema = self._infer_schema(df)

        # Step 4: Clean data
        cleaned_df, transform_log = self._clean_data(df, schema)

        # Step 5: Generate report
        report = self._generate_report(cleaned_df, schema, transform_log)

        # Step 6: Save outputs
        output_paths = self._save_outputs(cleaned_df, schema, transform_log, report)

        return output_paths

    # ========================================
    # VISMAY: TODO - Implement these methods
    # ========================================

    def _ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Detect file type and validate

        INPUT:
            - file_path: Path to file

        OUTPUT:
            - Dict {"type": "csv|excel|pdf|json|...", "size_bytes": int, "encoding": str}

        TODO: Implement file type detection using magic bytes + extension
        - Support CSV, Excel, JSON, PDF
        - Detect encoding
        - Return file metadata
        """
        print("[DataAgent] TODO: Implement _ingest_file()")

        # PLACEHOLDER: For now, assume CSV
        return {
            "type": "csv",
            "size_bytes": Path(file_path).stat().st_size,
            "encoding": "utf-8"
        }

    def _preprocess(self, file_path: str, file_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert file to DataFrame

        INPUT:
            - file_path: Path to file
            - file_info: File metadata from _ingest_file()

        OUTPUT:
            - pd.DataFrame: Preprocessed data

        TODO: Implement format conversion
        - CSV/Excel -> DataFrame
        - PDF -> OCR/text extraction -> structured data
        - JSON -> DataFrame
        """
        print("[DataAgent] TODO: Implement _preprocess()")

        # PLACEHOLDER: Read CSV
        df = pd.read_csv(file_path)
        return df

    def _infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infer column types, stats, and generate warnings

        INPUT:
            - df: DataFrame

        OUTPUT:
            - Dict {
                "columns": [
                    {"name": str, "type": str, "null_pct": float, "unique_count": int, ...}
                ],
                "warnings": [str]
              }

        Uses Claude to intelligently infer:
        - Semantic types (numeric, categorical, datetime, text, geo, etc.)
        - Data quality issues
        - Reasonable warnings and recommendations
        """
        print("[DataAgent] Inferring schema with Claude...")

        # Compute basic statistics
        basic_stats = {}
        for col in df.columns:
            col_data = df[col]
            stats = {
                "name": col,
                "dtype": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "null_pct": float(col_data.isnull().sum() / len(df)),
                "unique_count": int(col_data.nunique()),
                "sample_values": col_data.dropna().head(10).tolist()
            }

            # Add numeric stats if applicable
            if pd.api.types.is_numeric_dtype(col_data):
                stats["min"] = float(col_data.min()) if not col_data.empty else None
                stats["max"] = float(col_data.max()) if not col_data.empty else None
                stats["mean"] = float(col_data.mean()) if not col_data.empty else None
                stats["std"] = float(col_data.std()) if not col_data.empty else None

            basic_stats[col] = stats

        # Prepare prompt for Claude
        prompt = f"""Analyze this dataset schema and provide intelligent type inference and data quality warnings.

Dataset Info:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}

Column Statistics:
{json.dumps(basic_stats, indent=2, default=str)}

Please analyze each column and return a JSON object with this structure:
{{
  "columns": [
    {{
      "name": "column_name",
      "inferred_type": "numeric|categorical|datetime|text|geo|id|boolean",
      "semantic_meaning": "brief description of what this column represents",
      "data_quality": "good|warning|poor",
      "issues": ["list of specific issues if any"]
    }}
  ],
  "warnings": [
    "Overall data quality warnings or recommendations"
  ]
}}

Focus on:
1. Semantic type (not just pandas dtype) - is it truly categorical, an ID, a date string, etc.?
2. Data quality issues (high nulls, suspicious values, potential duplicates)
3. Actionable warnings

Return ONLY valid JSON, no markdown or explanations."""

        try:
            # Call Claude
            response = claude.call(prompt, max_tokens=2048)

            # Parse response
            # Remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:].strip()

            claude_analysis = json.loads(response_clean)

            # Merge Claude's analysis with our basic stats
            schema = {
                "columns": [],
                "warnings": claude_analysis.get("warnings", [])
            }

            for col_name in df.columns:
                # Find Claude's analysis for this column
                claude_col = next((c for c in claude_analysis.get("columns", []) if c["name"] == col_name), None)

                col_schema = {
                    "name": col_name,
                    "type": basic_stats[col_name]["dtype"],
                    "inferred_type": claude_col.get("inferred_type", "unknown") if claude_col else "unknown",
                    "semantic_meaning": claude_col.get("semantic_meaning", "") if claude_col else "",
                    "null_pct": basic_stats[col_name]["null_pct"],
                    "unique_count": basic_stats[col_name]["unique_count"],
                    "data_quality": claude_col.get("data_quality", "unknown") if claude_col else "unknown",
                    "issues": claude_col.get("issues", []) if claude_col else []
                }

                # Add numeric stats if available
                if "min" in basic_stats[col_name]:
                    col_schema["min"] = basic_stats[col_name]["min"]
                    col_schema["max"] = basic_stats[col_name]["max"]
                    col_schema["mean"] = basic_stats[col_name]["mean"]
                    col_schema["std"] = basic_stats[col_name]["std"]

                schema["columns"].append(col_schema)

            print(f"[DataAgent] ✓ Schema inferred: {len(schema['columns'])} columns, {len(schema['warnings'])} warnings")
            return schema

        except Exception as e:
            print(f"[DataAgent] ⚠️  Claude schema inference failed: {e}")
            print("[DataAgent] Falling back to basic schema")

            # Fallback to basic schema
            schema = {
                "columns": [],
                "warnings": [f"Claude inference failed: {str(e)}. Using basic schema."]
            }

            for col in df.columns:
                schema["columns"].append({
                    "name": col,
                    "type": str(df[col].dtype),
                    "null_pct": float(df[col].isnull().sum() / len(df)),
                    "unique_count": int(df[col].nunique())
                })

            return schema

    def _clean_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> tuple:
        """
        Apply transformations to clean data

        INPUT:
            - df: Raw DataFrame
            - schema: Schema from _infer_schema()

        OUTPUT:
            - (cleaned_df, transformation_log)
            - transformation_log: Dict with operations list

        Uses Claude to generate intelligent cleaning strategy based on schema
        """
        print("[DataAgent] Generating cleaning strategy with Claude...")

        original_rows = len(df)
        operations = []
        cleaned_df = df.copy()

        # Prepare schema summary for Claude
        schema_summary = []
        for col in schema["columns"]:
            schema_summary.append({
                "name": col["name"],
                "type": col.get("type", "unknown"),
                "inferred_type": col.get("inferred_type", "unknown"),
                "null_pct": col.get("null_pct", 0),
                "unique_count": col.get("unique_count", 0),
                "issues": col.get("issues", []),
                "data_quality": col.get("data_quality", "unknown")
            })

        # Ask Claude for cleaning strategy
        prompt = f"""Based on this dataset schema, recommend a data cleaning strategy.

Schema Analysis:
{json.dumps(schema_summary, indent=2)}

Dataset Warnings:
{json.dumps(schema.get("warnings", []), indent=2)}

Total rows: {len(df)}

Please provide a cleaning strategy as a JSON object:
{{
  "operations": [
    {{
      "operation": "drop_column|impute_missing|cast_type|remove_duplicates|handle_outliers|normalize_categorical",
      "column": "column_name (or null for row-level ops)",
      "reason": "why this operation is needed",
      "parameters": {{"method": "median|mode|drop", "threshold": 0.5}}
    }}
  ],
  "recommendations": ["additional recommendations"]
}}

Cleaning rules:
1. Drop columns with >50% missing values
2. For <50% missing: impute numeric (median), categorical (mode)
3. Cast datetime strings to proper types if detected
4. Remove exact duplicate rows
5. Handle outliers in numeric columns (>3 std devs) by capping
6. Normalize categorical values (lowercase, strip whitespace)

Return ONLY valid JSON."""

        try:
            # Call Claude
            response = claude.call(prompt, max_tokens=2048)

            # Parse response
            response_clean = response.strip()
            if response_clean.startswith("```"):
                lines = response_clean.split('\n')
                response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean
                if response_clean.startswith("json"):
                    response_clean = response_clean[4:].strip()

            cleaning_strategy = json.loads(response_clean)

            # Execute cleaning operations
            for op in cleaning_strategy.get("operations", []):
                operation_type = op.get("operation")
                column = op.get("column")
                reason = op.get("reason", "")
                params = op.get("parameters", {})

                try:
                    if operation_type == "drop_column" and column in cleaned_df.columns:
                        cleaned_df = cleaned_df.drop(columns=[column])
                        operations.append({
                            "operation": "drop_column",
                            "column": column,
                            "reason": reason,
                            "rows_affected": 0
                        })

                    elif operation_type == "impute_missing" and column in cleaned_df.columns:
                        method = params.get("method", "median")
                        missing_count = cleaned_df[column].isnull().sum()

                        if missing_count > 0:
                            if method == "median" and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                            elif method == "mode":
                                mode_val = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else None
                                if mode_val is not None:
                                    cleaned_df[column].fillna(mode_val, inplace=True)
                            elif method == "drop":
                                cleaned_df = cleaned_df.dropna(subset=[column])

                            operations.append({
                                "operation": "impute_missing",
                                "column": column,
                                "method": method,
                                "reason": reason,
                                "rows_affected": int(missing_count)
                            })

                    elif operation_type == "cast_type" and column in cleaned_df.columns:
                        target_type = params.get("target_type", "datetime")
                        if target_type == "datetime":
                            cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                            operations.append({
                                "operation": "cast_type",
                                "column": column,
                                "target_type": target_type,
                                "reason": reason,
                                "rows_affected": 0
                            })

                    elif operation_type == "remove_duplicates":
                        before_count = len(cleaned_df)
                        cleaned_df = cleaned_df.drop_duplicates()
                        removed = before_count - len(cleaned_df)
                        if removed > 0:
                            operations.append({
                                "operation": "remove_duplicates",
                                "column": None,
                                "reason": reason,
                                "rows_affected": int(removed)
                            })

                    elif operation_type == "handle_outliers" and column in cleaned_df.columns:
                        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                            mean = cleaned_df[column].mean()
                            std = cleaned_df[column].std()
                            threshold = params.get("std_threshold", 3)

                            upper_cap = mean + (threshold * std)
                            lower_cap = mean - (threshold * std)

                            outliers = ((cleaned_df[column] > upper_cap) | (cleaned_df[column] < lower_cap)).sum()

                            if outliers > 0:
                                cleaned_df[column] = cleaned_df[column].clip(lower=lower_cap, upper=upper_cap)
                                operations.append({
                                    "operation": "handle_outliers",
                                    "column": column,
                                    "method": "cap_at_3std",
                                    "reason": reason,
                                    "rows_affected": int(outliers)
                                })

                    elif operation_type == "normalize_categorical" and column in cleaned_df.columns:
                        if cleaned_df[column].dtype == 'object':
                            cleaned_df[column] = cleaned_df[column].str.strip().str.lower()
                            operations.append({
                                "operation": "normalize_categorical",
                                "column": column,
                                "reason": reason,
                                "rows_affected": len(cleaned_df)
                            })

                except Exception as op_error:
                    print(f"[DataAgent] ⚠️  Failed to execute {operation_type} on {column}: {op_error}")

            print(f"[DataAgent] ✓ Cleaning complete: {len(operations)} operations applied")

        except Exception as e:
            print(f"[DataAgent] ⚠️  Claude cleaning strategy failed: {e}")
            print("[DataAgent] Applying basic cleaning...")

            # Basic fallback cleaning
            # 1. Remove duplicates
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            if before_count > len(cleaned_df):
                operations.append({
                    "operation": "remove_duplicates",
                    "column": None,
                    "reason": "Basic cleaning",
                    "rows_affected": before_count - len(cleaned_df)
                })

        # Create transformation log
        transform_log = {
            "operations": operations,
            "provenance": {
                "original_rows": original_rows,
                "final_rows": len(cleaned_df),
                "columns_dropped": [op["column"] for op in operations if op["operation"] == "drop_column"],
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }

        return cleaned_df, transform_log

    def _generate_report(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        transform_log: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable data quality report

        INPUT:
            - df: Cleaned DataFrame
            - schema: Schema dict
            - transform_log: Transformation log

        OUTPUT:
            - str: Markdown report

        Uses Claude to generate an insightful, actionable report
        """
        print("[DataAgent] Generating report with Claude...")

        # Prepare summary information
        provenance = transform_log.get("provenance", {})
        operations = transform_log.get("operations", [])

        summary_info = {
            "original_rows": provenance.get("original_rows", len(df)),
            "final_rows": len(df),
            "total_columns": len(df.columns),
            "columns_dropped": provenance.get("columns_dropped", []),
            "operations_count": len(operations),
            "warnings": schema.get("warnings", [])
        }

        # Column summary
        column_summary = []
        for col in schema["columns"]:
            column_summary.append({
                "name": col["name"],
                "type": col.get("inferred_type", col.get("type", "unknown")),
                "data_quality": col.get("data_quality", "unknown"),
                "null_pct": col.get("null_pct", 0),
                "issues": col.get("issues", [])
            })

        # Operations summary
        operations_summary = []
        for op in operations:
            operations_summary.append({
                "operation": op.get("operation"),
                "column": op.get("column", "N/A"),
                "reason": op.get("reason", ""),
                "rows_affected": op.get("rows_affected", 0)
            })

        # Build prompt for Claude
        prompt = f"""Generate a comprehensive data quality report in Markdown format.

Dataset Summary:
{json.dumps(summary_info, indent=2)}

Column Analysis:
{json.dumps(column_summary, indent=2)}

Transformations Applied:
{json.dumps(operations_summary, indent=2)}

Please create a professional data quality report with these sections:

1. **Executive Summary** - High-level overview of the dataset and data quality
2. **Dataset Overview** - Key statistics (rows, columns, transformations)
3. **Data Quality Assessment** - Analysis of each column's quality with specific issues
4. **Transformations Applied** - Detailed log of cleaning operations and their impact
5. **Warnings & Issues** - Critical data quality problems that need attention
6. **Recommendations** - Actionable next steps for further improvement

Make it:
- Professional and actionable
- Highlight critical issues clearly
- Include specific column names and statistics
- Provide concrete recommendations

Return ONLY the markdown report, no JSON or code blocks."""

        try:
            # Call Claude
            response = claude.call(prompt, max_tokens=3096)

            # Clean up response (remove any code block markers)
            report = response.strip()
            if report.startswith("```markdown"):
                report = report[11:].strip()
            elif report.startswith("```"):
                report = '\n'.join(report.split('\n')[1:-1])

            # Ensure it ends properly
            if report.endswith("```"):
                report = report[:-3].strip()

            print("[DataAgent] ✓ Report generated")
            return report

        except Exception as e:
            print(f"[DataAgent] ⚠️  Claude report generation failed: {e}")
            print("[DataAgent] Generating basic report...")

            # Fallback to basic report
            report = f"""# Data Quality Report

## Executive Summary
Dataset processed with {summary_info['operations_count']} cleaning operations applied.

## Dataset Overview
- **Original rows**: {summary_info['original_rows']}
- **Final rows**: {summary_info['final_rows']}
- **Total columns**: {summary_info['total_columns']}
- **Columns dropped**: {len(summary_info['columns_dropped'])}

## Data Quality Assessment

"""
            for col in column_summary:
                report += f"### {col['name']}\n"
                report += f"- **Type**: {col['type']}\n"
                report += f"- **Quality**: {col['data_quality']}\n"
                report += f"- **Missing**: {col['null_pct']:.1%}\n"
                if col['issues']:
                    report += f"- **Issues**: {', '.join(col['issues'])}\n"
                report += "\n"

            report += "## Transformations Applied\n\n"
            for i, op in enumerate(operations_summary, 1):
                report += f"{i}. **{op['operation']}** on `{op['column']}`: {op['reason']}\n"
                report += f"   - Rows affected: {op['rows_affected']}\n\n"

            if summary_info['warnings']:
                report += "## Warnings\n\n"
                for warning in summary_info['warnings']:
                    report += f"- {warning}\n"

            report += "\n## Recommendations\n\n"
            report += "- Review the cleaning operations above\n"
            report += "- Verify that imputation strategies are appropriate\n"
            report += "- Check for any remaining data quality issues\n"

            return report

    def _save_outputs(
        self,
        cleaned_df: pd.DataFrame,
        schema: Dict[str, Any],
        transform_log: Dict[str, Any],
        report: str
    ) -> Dict[str, str]:
        """
        Save all outputs to files

        INPUT:
            - cleaned_df: Cleaned DataFrame
            - schema: Schema dict
            - transform_log: Transformation log
            - report: Report markdown

        OUTPUT:
            - Dict with paths to saved files
        """
        # Create directories
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        DATA_QUALITY_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned data
        cleaned_path = CLEANED_DIR / "cleaned_data.csv"
        cleaned_df.to_csv(cleaned_path, index=False)

        # Save schema
        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema, f, indent=2)

        # Save transformation log
        with open(TRANSFORMATION_LOG_FILE, "w") as f:
            json.dump(transform_log, f, indent=2)

        # Save report
        with open(DATA_QUALITY_REPORT_FILE, "w") as f:
            f.write(report)

        print(f"[DataAgent] Saved outputs:")
        print(f"  - Cleaned data: {cleaned_path}")
        print(f"  - Schema: {SCHEMA_FILE}")
        print(f"  - Transform log: {TRANSFORMATION_LOG_FILE}")
        print(f"  - Report: {DATA_QUALITY_REPORT_FILE}")

        return {
            "cleaned_data_path": str(cleaned_path),
            "schema_path": str(SCHEMA_FILE),
            "transformation_log_path": str(TRANSFORMATION_LOG_FILE),
            "report_path": str(DATA_QUALITY_REPORT_FILE),
            "stats": {
                "rows": len(cleaned_df),
                "columns": len(cleaned_df.columns)
            }
        }
