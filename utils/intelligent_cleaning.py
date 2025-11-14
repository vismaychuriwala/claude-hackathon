"""
Intelligent Data Cleaning System

Uses Claude to generate flexible data cleaning code that executes in sandbox.
"""

import json
from typing import Dict, Any, List, Optional
import pandas as pd
from utils.claude_client import ClaudeClient
from utils.sandbox import SandboxExecutor


class IntelligentDataCleaner:
    """
    Uses Claude to intelligently plan and execute data cleaning operations.
    """

    def __init__(self, agent_name: str = "data"):
        """
        Initialize intelligent data cleaner.

        Args:
            agent_name: Name of the agent for Claude client configuration
        """
        self.claude = ClaudeClient(agent_name=agent_name)
        self.sandbox = SandboxExecutor(timeout=60)

    def plan_cleaning(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Claude to plan appropriate data cleaning operations.

        Args:
            df: DataFrame to clean
            schema: Inferred schema with data quality information

        Returns:
            Cleaning plan with operations and code
        """
        # Analyze data quality issues
        quality_issues = self._detect_quality_issues(df, schema)

        # Create prompt for Claude
        prompt = self._create_cleaning_prompt(df, schema, quality_issues)

        # Get Claude's recommendation
        response = self.claude.call(
            prompt=prompt,
            max_tokens=4096
        )

        # Parse response
        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean.split('```json')[1].split('```')[0].strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.split('```')[1].split('```')[0].strip()

            plan = json.loads(response_clean)
            return plan
        except json.JSONDecodeError as e:
            print(f"Failed to parse Claude's cleaning plan: {e}")
            return self._create_fallback_plan(quality_issues)

    def _detect_quality_issues(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect data quality issues that need cleaning.
        """
        issues = []

        for col in df.columns:
            col_data = df[col]

            # Missing values
            missing_count = col_data.isna().sum()
            if missing_count > 0:
                issues.append({
                    'column': col,
                    'type': 'missing_values',
                    'severity': 'high' if missing_count / len(df) > 0.3 else 'medium',
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                })

            # Duplicate values in potential ID columns
            if col_data.nunique() > len(df) * 0.9:  # High cardinality, might be ID
                dup_count = col_data.duplicated().sum()
                if dup_count > 0:
                    issues.append({
                        'column': col,
                        'type': 'duplicates_in_id_column',
                        'severity': 'high',
                        'count': int(dup_count)
                    })

            # Type inconsistencies
            if col_data.dtype == 'object':
                # Check if should be numeric
                try:
                    pd.to_numeric(col_data.dropna(), errors='raise')
                    issues.append({
                        'column': col,
                        'type': 'type_inconsistency',
                        'severity': 'medium',
                        'suggested_type': 'numeric',
                        'current_type': 'object'
                    })
                except:
                    pass

                # Check if should be datetime
                try:
                    pd.to_datetime(col_data.dropna(), errors='raise')
                    issues.append({
                        'column': col,
                        'type': 'type_inconsistency',
                        'severity': 'medium',
                        'suggested_type': 'datetime',
                        'current_type': 'object'
                    })
                except:
                    pass

            # Outliers for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                col_clean = col_data.dropna()
                if len(col_clean) > 0:
                    Q1 = col_clean.quantile(0.25)
                    Q3 = col_clean.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = col_clean[(col_clean < Q1 - 3 * IQR) | (col_clean > Q3 + 3 * IQR)]
                    if len(outliers) > 0:
                        issues.append({
                            'column': col,
                            'type': 'outliers',
                            'severity': 'low',
                            'count': int(len(outliers)),
                            'percentage': float(len(outliers) / len(col_clean) * 100)
                        })

        # Duplicate rows
        dup_rows = df.duplicated().sum()
        if dup_rows > 0:
            issues.append({
                'column': 'all',
                'type': 'duplicate_rows',
                'severity': 'high',
                'count': int(dup_rows)
            })

        return issues

    def _create_cleaning_prompt(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        quality_issues: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for cleaning planning"""
        # Sample data for context
        sample_data = df.head(5).to_dict(orient='records')

        return f"""You are an expert data cleaning specialist. Based on the dataset information and detected quality issues, generate Python code to clean the data appropriately.

DATASET INFO:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}

SCHEMA INFORMATION:
{json.dumps(schema, indent=2, default=str)}

DETECTED QUALITY ISSUES:
{json.dumps(quality_issues, indent=2)}

SAMPLE DATA (first 5 rows):
{json.dumps(sample_data, indent=2, default=str)}

INSTRUCTIONS:
1. Review the quality issues carefully
2. For each issue, decide on an appropriate cleaning strategy:
   - Missing values: Impute (mean/median/mode/forward-fill) or drop based on severity and data type
   - Type inconsistencies: Cast to appropriate type with error handling
   - Duplicates: Remove or flag based on whether they're in ID columns or full rows
   - Outliers: Cap, transform, or document (don't blindly remove)

3. Generate Python code that:
   - Operates on a DataFrame variable named 'df'
   - Returns the cleaned DataFrame as 'cleaned_df'
   - Creates a transformation log as 'transform_log' (list of dicts with operation details)
   - Includes comments explaining each transformation
   - Handles errors gracefully
   - Preserves data integrity

4. Return your response as a JSON object:
{{
  "cleaning_operations": [
    {{
      "operation": "impute_missing",
      "column": "age",
      "method": "median",
      "rationale": "Median is robust to outliers for age data"
    }}
  ],
  "cleaning_code": "# Python code here\\ncleaned_df = df.copy()\\ntransform_log = []",
  "expected_improvements": {{
    "missing_values_reduced": 150,
    "duplicates_removed": 10,
    "type_conversions": 2
  }},
  "warnings": [
    "Dropping 5% of rows due to critical missing values in key columns"
  ]
}}

IMPORTANT:
- DO NOT blindly remove data
- Prefer imputation over deletion when reasonable
- Document all transformations in transform_log
- Be cautious with outlier removal (cap instead of remove when possible)

Return ONLY the JSON object, no additional text.
"""

    def _create_fallback_plan(self, quality_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a basic fallback cleaning plan"""
        operations = []
        code_parts = [
            "cleaned_df = df.copy()",
            "transform_log = []"
        ]

        # Simple duplicate removal
        if any(issue['type'] == 'duplicate_rows' for issue in quality_issues):
            operations.append({
                'operation': 'remove_duplicates',
                'rationale': 'Fallback: Remove duplicate rows'
            })
            code_parts.append("""
# Remove duplicate rows
initial_rows = len(cleaned_df)
cleaned_df = cleaned_df.drop_duplicates()
removed_rows = initial_rows - len(cleaned_df)
if removed_rows > 0:
    transform_log.append({
        'operation': 'remove_duplicates',
        'rows_removed': removed_rows
    })
""")

        # Simple missing value handling
        missing_issues = [i for i in quality_issues if i['type'] == 'missing_values']
        for issue in missing_issues[:3]:  # Limit to first 3
            col = issue['column']
            operations.append({
                'operation': 'drop_missing',
                'column': col,
                'rationale': 'Fallback: Drop missing values'
            })
            code_parts.append(f"""
# Drop missing values in {col}
cleaned_df = cleaned_df.dropna(subset=['{col}'])
transform_log.append({{'operation': 'drop_missing', 'column': '{col}'}})
""")

        return {
            'cleaning_operations': operations,
            'cleaning_code': '\n'.join(code_parts),
            'expected_improvements': {},
            'warnings': ['Using fallback cleaning plan due to Claude parsing error']
        }

    def execute_cleaning(
        self,
        df: pd.DataFrame,
        cleaning_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute cleaning operations in sandbox.

        Args:
            df: DataFrame to clean
            cleaning_plan: Plan from plan_cleaning()

        Returns:
            Dictionary with cleaned DataFrame and transformation log
        """
        code = cleaning_plan.get('cleaning_code', '')

        print("[IntelligentDataCleaner] Executing cleaning code in sandbox...")

        # Execute in sandbox
        success, result_dict, stdout, stderr = self.sandbox.execute(
            code=code,
            data_context={'df': df},
            return_variable=None  # We'll get both cleaned_df and transform_log from locals
        )

        if not success:
            return {
                'success': False,
                'error': stderr,
                'cleaned_df': df,  # Return original
                'transform_log': []
            }

        # Extract cleaned_df and transform_log from execution context
        # Since we can't directly return multiple variables, we need to modify the approach
        # Let's execute code that creates a result dict
        code_with_result = f"""
{code}

# Create result dictionary
result = {{
    'cleaned_df_shape': cleaned_df.shape,
    'transform_log': transform_log
}}
"""

        success, result, stdout, stderr = self.sandbox.execute(
            code=code_with_result,
            data_context={'df': df},
            return_variable='result'
        )

        # Now execute again to get the actual cleaned_df
        # This is a workaround since we need to return the DataFrame itself
        code_final = f"""
{code}
cleaned_df = cleaned_df
"""

        success_df, cleaned_df, stdout_df, stderr_df = self.sandbox.execute(
            code=code_final,
            data_context={'df': df},
            return_variable='cleaned_df'
        )

        if success and success_df:
            return {
                'success': True,
                'cleaned_df': cleaned_df,
                'transform_log': result.get('transform_log', []),
                'expected_improvements': cleaning_plan.get('expected_improvements', {}),
                'warnings': cleaning_plan.get('warnings', [])
            }
        else:
            return {
                'success': False,
                'error': stderr or stderr_df,
                'cleaned_df': df,
                'transform_log': []
            }

    def validate_cleaning(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        transform_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that cleaning operations produced reasonable results.

        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            transform_log: Log of transformations

        Returns:
            Validation report
        """
        validation = {
            'shape_change': {
                'original': original_df.shape,
                'cleaned': cleaned_df.shape,
                'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
                'columns_removed': original_df.shape[1] - cleaned_df.shape[1]
            },
            'data_loss_percentage': float((original_df.shape[0] - cleaned_df.shape[0]) / original_df.shape[0] * 100),
            'issues': []
        }

        # Check for excessive data loss
        if validation['data_loss_percentage'] > 50:
            validation['issues'].append({
                'severity': 'high',
                'message': f"Excessive data loss: {validation['data_loss_percentage']:.1f}% of rows removed"
            })

        # Check for missing required columns
        missing_cols = set(original_df.columns) - set(cleaned_df.columns)
        if missing_cols:
            validation['issues'].append({
                'severity': 'high',
                'message': f"Columns removed: {missing_cols}"
            })

        # Check for new missing values
        original_missing = original_df.isna().sum().sum()
        cleaned_missing = cleaned_df.isna().sum().sum()
        if cleaned_missing > original_missing:
            validation['issues'].append({
                'severity': 'medium',
                'message': f"Cleaning introduced new missing values: {cleaned_missing - original_missing}"
            })

        validation['status'] = 'pass' if len(validation['issues']) == 0 else 'warning'

        return validation
