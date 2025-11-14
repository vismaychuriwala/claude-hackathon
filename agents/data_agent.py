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

        TODO: Use Claude to infer schema intelligently
        - Detect numeric, categorical, datetime, text, geo columns
        - Compute null%, min/max, mean, unique counts
        - Generate warnings for data quality issues
        - Write to schema.json
        """
        print("[DataAgent] TODO: Implement _infer_schema() using Claude")

        # PLACEHOLDER: Basic schema
        schema = {
            "columns": [],
            "warnings": []
        }

        for col in df.columns:
            schema["columns"].append({
                "name": col,
                "type": str(df[col].dtype),
                "null_pct": df[col].isnull().sum() / len(df),
                "unique_count": df[col].nunique()
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

        TODO: Use Claude to generate cleaning strategy
        - Type casting
        - Handle missing values (drop if null>50%, else impute)
        - Deduplicate rows
        - Outlier detection (IQR/z-score)
        - Categorical normalization
        - Log all transformations
        """
        print("[DataAgent] TODO: Implement _clean_data() using Claude")

        # PLACEHOLDER: Return original data
        transform_log = {
            "operations": [],
            "provenance": {
                "original_rows": len(df),
                "final_rows": len(df),
                "columns_dropped": []
            }
        }

        return df, transform_log

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

        TODO: Use Claude to generate insightful report
        - Summary statistics
        - Data quality warnings
        - Transformation summary
        - Recommendations
        """
        print("[DataAgent] TODO: Implement _generate_report() using Claude")

        # PLACEHOLDER: Basic report
        report = f"""# Data Quality Report

## Summary
- Total rows: {len(df)}
- Total columns: {len(df.columns)}

## TODO: Add detailed analysis using Claude
"""
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
