"""
Intelligent Analysis Planning System

Uses Claude to intelligently determine which analyses are appropriate
for a given dataset, generate code, and validate results.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from utils.claude_client import ClaudeClient
from utils.sandbox import execute_analysis_code, CodeValidationResult


class DatasetProfile:
    """Comprehensive profile of a dataset for intelligent analysis planning"""

    def __init__(self, df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None):
        """
        Create dataset profile.

        Args:
            df: DataFrame to profile
            schema: Optional pre-computed schema from Data Agent
        """
        self.df = df
        self.schema = schema or {}
        self._generate_profile()

    def _generate_profile(self):
        """Generate comprehensive dataset profile"""
        self.profile = {
            'shape': {
                'rows': len(self.df),
                'columns': len(self.df.columns)
            },
            'columns': self._profile_columns(),
            'data_quality': self._assess_data_quality(),
            'relationships': self._detect_relationships(),
            'business_context': self._infer_business_context()
        }

    def _profile_columns(self) -> List[Dict[str, Any]]:
        """Profile each column in detail"""
        columns_info = []

        for col in self.df.columns:
            col_data = self.df[col]
            dtype = str(col_data.dtype)

            col_info = {
                'name': col,
                'dtype': dtype,
                'missing_count': int(col_data.isna().sum()),
                'missing_pct': float(col_data.isna().sum() / len(col_data) * 100),
                'unique_count': int(col_data.nunique()),
                'cardinality': 'high' if col_data.nunique() > len(col_data) * 0.5 else 'low'
            }

            # Semantic type from schema if available
            if self.schema and 'columns' in self.schema:
                schema_col = next((c for c in self.schema['columns'] if c['name'] == col), None)
                if schema_col:
                    col_info['semantic_type'] = schema_col.get('semantic_type', 'unknown')

            # Type-specific profiling
            if pd.api.types.is_numeric_dtype(col_data):
                col_info['numeric_profile'] = self._profile_numeric(col_data)
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_info['datetime_profile'] = self._profile_datetime(col_data)
            else:
                col_info['categorical_profile'] = self._profile_categorical(col_data)

            columns_info.append(col_info)

        return columns_info

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric column"""
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return {'error': 'all_missing'}

        # Check if likely categorical (despite numeric type)
        unique_ratio = series_clean.nunique() / len(series_clean)
        if unique_ratio < 0.05 or series_clean.nunique() < 20:
            return {
                'likely_categorical': True,
                'unique_values': int(series_clean.nunique()),
                'value_counts': series_clean.value_counts().head(10).to_dict()
            }

        # Check if likely identifier
        if series_clean.is_monotonic_increasing or unique_ratio > 0.95:
            return {
                'likely_identifier': True,
                'unique_ratio': float(unique_ratio)
            }

        # True numeric analysis
        return {
            'min': float(series_clean.min()),
            'max': float(series_clean.max()),
            'range': float(series_clean.max() - series_clean.min()),
            'has_negative': bool((series_clean < 0).any()),
            'has_decimals': bool((series_clean % 1 != 0).any()),
            'zero_count': int((series_clean == 0).sum()),
            'distribution_shape': self._assess_distribution(series_clean)
        }

    def _assess_distribution(self, series: pd.Series) -> str:
        """Assess distribution shape of numeric data"""
        if len(series) < 3:
            return 'insufficient_data'

        from scipy.stats import skew, kurtosis

        skewness = skew(series)
        kurt = kurtosis(series)

        if abs(skewness) < 0.5 and abs(kurt) < 1:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif abs(kurt) > 3:
            return 'heavy_tailed'
        else:
            return 'irregular'

    def _profile_datetime(self, series: pd.Series) -> Dict[str, Any]:
        """Profile datetime column"""
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return {'error': 'all_missing'}

        return {
            'min_date': str(series_clean.min()),
            'max_date': str(series_clean.max()),
            'date_range_days': (series_clean.max() - series_clean.min()).days,
            'is_sorted': bool(series_clean.is_monotonic_increasing or series_clean.is_monotonic_decreasing),
            'has_duplicates': bool(series_clean.duplicated().any())
        }

    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical/text column"""
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return {'error': 'all_missing'}

        value_counts = series_clean.value_counts()

        return {
            'unique_count': int(series_clean.nunique()),
            'top_values': value_counts.head(10).to_dict(),
            'max_length': int(series_clean.astype(str).str.len().max()),
            'avg_length': float(series_clean.astype(str).str.len().mean())
        }

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality"""
        return {
            'total_missing': int(self.df.isna().sum().sum()),
            'missing_pct': float(self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'complete_rows': int((~self.df.isna().any(axis=1)).sum())
        }

    def _detect_relationships(self) -> Dict[str, Any]:
        """Detect potential relationships between columns"""
        relationships = {
            'potential_time_series': [],
            'potential_groupings': [],
            'potential_hierarchies': []
        }

        # Detect time series (datetime + numeric columns)
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if datetime_cols and numeric_cols:
            relationships['potential_time_series'] = [
                {'time_col': dt, 'value_cols': numeric_cols}
                for dt in datetime_cols
            ]

        # Detect groupings (low cardinality + numeric columns)
        for col in self.df.columns:
            if self.df[col].nunique() < 20 and self.df[col].nunique() > 1:
                if numeric_cols:
                    relationships['potential_groupings'].append({
                        'group_by': col,
                        'aggregate_cols': numeric_cols
                    })

        return relationships

    def _infer_business_context(self) -> Dict[str, Any]:
        """Infer business context from column names and data"""
        context = {
            'domain_indicators': [],
            'suggested_analyses': []
        }

        col_names_lower = [col.lower() for col in self.df.columns]

        # Detect financial data
        financial_keywords = ['price', 'amount', 'revenue', 'cost', 'profit', 'sales', 'balance']
        if any(keyword in ' '.join(col_names_lower) for keyword in financial_keywords):
            context['domain_indicators'].append('financial')
            context['suggested_analyses'].append('financial_metrics')

        # Detect user/customer data
        user_keywords = ['user', 'customer', 'client', 'account', 'member']
        if any(keyword in ' '.join(col_names_lower) for keyword in user_keywords):
            context['domain_indicators'].append('customer_analytics')
            context['suggested_analyses'].append('cohort_analysis')

        # Detect temporal data
        if self.df.select_dtypes(include=['datetime64']).columns.any():
            context['domain_indicators'].append('time_series')
            context['suggested_analyses'].append('trend_analysis')

        return context

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return self.profile


class IntelligentAnalyzer:
    """
    Uses Claude to intelligently plan and execute analyses based on dataset characteristics.
    """

    def __init__(self, agent_name: str = "analysis"):
        """
        Initialize intelligent analyzer.

        Args:
            agent_name: Name of the agent for Claude client configuration
        """
        self.claude = ClaudeClient(agent_name=agent_name)

    def plan_analysis(
        self,
        df: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None,
        analysis_goal: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Use Claude to plan appropriate analyses for the dataset.

        Args:
            df: DataFrame to analyze
            schema: Optional schema from Data Agent
            analysis_goal: Type of analysis (comprehensive, quick, specific)

        Returns:
            Analysis plan with recommended analyses and code
        """
        # Generate dataset profile
        profile = DatasetProfile(df, schema)
        profile_dict = profile.to_dict()

        # Create prompt for Claude
        prompt = self._create_planning_prompt(profile_dict, analysis_goal)

        # Get Claude's recommendation
        response = self.claude.call(
            prompt=prompt,
            max_tokens=4096
        )

        # Parse response
        try:
            # Extract JSON from response (Claude may wrap it in markdown)
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean.split('```json')[1].split('```')[0].strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.split('```')[1].split('```')[0].strip()

            plan = json.loads(response_clean)
            return plan
        except json.JSONDecodeError as e:
            # Fallback: create basic plan
            print(f"Failed to parse Claude's analysis plan: {e}")
            return self._create_fallback_plan(profile_dict)

    def _create_planning_prompt(self, profile: Dict[str, Any], goal: str) -> str:
        """Create prompt for analysis planning"""
        return f"""You are an expert data analyst. Based on the dataset profile below, recommend appropriate statistical analyses and generate Python code to perform them.

DATASET PROFILE:
{json.dumps(profile, indent=2)}

ANALYSIS GOAL: {goal}

INSTRUCTIONS:
1. Review the dataset profile carefully, paying attention to:
   - Column data types and semantic meaning
   - Whether numeric columns are truly quantitative or categorical/identifiers
   - Data quality issues
   - Potential relationships between columns
   - Business context indicators

2. Recommend ONLY analyses that are appropriate for this specific dataset.
   - DO NOT compute mean/median for identifier columns (IDs, ZIP codes, etc.)
   - DO NOT compute mean/median for categorical data encoded as numbers
   - DO recommend appropriate analyses for the actual data type (mode for categorical, trends for time series, etc.)

3. For each recommended analysis, generate Python code that:
   - Uses pandas (pd), numpy (np), and scipy.stats (scipy_stats)
   - Operates on the DataFrame variable named 'df'
   - Stores the result in a dictionary variable named 'analysis_result'
   - Includes appropriate error handling
   - Is efficient and well-commented

4. Return your response as a JSON object with this structure:
{{
  "recommended_analyses": [
    {{
      "name": "descriptive_statistics_for_sales",
      "description": "Compute descriptive statistics for the sales amount column",
      "rationale": "Sales amount is a true quantitative variable suitable for mean/median/std",
      "code": "# Python code here\\nanalysis_result = {{'mean': df['sales'].mean()}}",
      "expected_output_type": "dict"
    }}
  ],
  "inappropriate_analyses": [
    {{
      "name": "mean_of_user_id",
      "reason": "user_id is an identifier, not a quantitative variable. Mean is not meaningful."
    }}
  ],
  "data_quality_warnings": [
    "Column X has 30% missing values"
  ]
}}

Return ONLY the JSON object, no additional text.
"""

    def _create_fallback_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback plan if Claude fails"""
        analyses = []

        # Find truly numeric columns (exclude identifiers)
        for col_info in profile.get('columns', []):
            if 'numeric_profile' in col_info:
                numeric_profile = col_info['numeric_profile']

                # Skip if likely categorical or identifier
                if numeric_profile.get('likely_categorical') or numeric_profile.get('likely_identifier'):
                    continue

                col_name = col_info['name']
                analyses.append({
                    'name': f'descriptive_stats_{col_name}',
                    'description': f'Basic statistics for {col_name}',
                    'rationale': 'Fallback analysis',
                    'code': f"""
# Descriptive statistics for {col_name}
col_data = df['{col_name}'].dropna()
analysis_result = {{
    'column': '{col_name}',
    'count': int(len(col_data)),
    'mean': float(col_data.mean()),
    'median': float(col_data.median()),
    'std': float(col_data.std())
}}
""",
                    'expected_output_type': 'dict'
                })

        return {
            'recommended_analyses': analyses,
            'inappropriate_analyses': [],
            'data_quality_warnings': ['Using fallback plan due to Claude parsing error']
        }

    def execute_analysis(
        self,
        df: pd.DataFrame,
        analysis_plan: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute the planned analyses in sandbox.

        Args:
            df: DataFrame to analyze
            analysis_plan: Plan from plan_analysis()
            timeout: Timeout per analysis in seconds

        Returns:
            Dictionary with execution results
        """
        results = {
            'successful_analyses': [],
            'failed_analyses': [],
            'execution_summary': {}
        }

        for analysis in analysis_plan.get('recommended_analyses', []):
            code = analysis['code']
            name = analysis['name']

            print(f"Executing analysis: {name}")

            # Execute in sandbox
            exec_result = execute_analysis_code(code, df, timeout=timeout)

            if exec_result.success:
                results['successful_analyses'].append({
                    'name': name,
                    'description': analysis['description'],
                    'result': exec_result.result,
                    'execution_time': exec_result.execution_time
                })
            else:
                results['failed_analyses'].append({
                    'name': name,
                    'description': analysis['description'],
                    'error': exec_result.stderr,
                    'validation_error': exec_result.validation_error
                })

        results['execution_summary'] = {
            'total_planned': len(analysis_plan.get('recommended_analyses', [])),
            'successful': len(results['successful_analyses']),
            'failed': len(results['failed_analyses'])
        }

        return results

    def validate_results(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Claude to validate that the executed analyses produced reasonable results.

        Args:
            df: Original DataFrame
            profile: Dataset profile
            execution_results: Results from execute_analysis()

        Returns:
            Validation report with feedback
        """
        prompt = f"""You are validating the results of automated data analysis.

DATASET PROFILE:
{json.dumps(profile, indent=2)}

EXECUTION RESULTS:
{json.dumps(execution_results, indent=2, default=str)}

Please review the results and provide validation feedback:

1. Are the results reasonable given the data characteristics?
2. Are there any surprising or suspicious values that might indicate errors?
3. Are there any important analyses that were missed?
4. Should any analyses be refined or re-run with different parameters?

Return a JSON object with this structure:
{{
  "validation_status": "pass" or "needs_refinement",
  "feedback": [
    {{
      "analysis": "name of analysis",
      "status": "valid" or "suspicious" or "incorrect",
      "comment": "explanation"
    }}
  ],
  "suggested_refinements": [
    {{
      "analysis": "name of analysis to refine",
      "reason": "why refinement is needed",
      "suggested_change": "what to change"
    }}
  ],
  "missing_analyses": [
    "description of important missing analysis"
  ]
}}

Return ONLY the JSON object.
"""

        response = self.claude.call(prompt=prompt, max_tokens=2048)

        try:
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean.split('```json')[1].split('```')[0].strip()
            elif response_clean.startswith('```'):
                response_clean = response_clean.split('```')[1].split('```')[0].strip()

            validation = json.loads(response_clean)
            return validation
        except json.JSONDecodeError:
            return {
                'validation_status': 'pass',
                'feedback': [],
                'suggested_refinements': [],
                'missing_analyses': [],
                'error': 'Failed to parse validation response'
            }
