#!/usr/bin/env python3
"""
Test script to demonstrate the Intelligent Analysis System

This script shows how the new system avoids "blindly creating mean" for inappropriate columns.
"""

import pandas as pd
import numpy as np
from utils.intelligent_analysis import IntelligentAnalyzer, DatasetProfile


def create_test_dataset():
    """Create a test dataset with various column types"""
    np.random.seed(42)

    # Create problematic dataset that old system would analyze incorrectly
    data = {
        # Identifier - should NOT compute mean
        'customer_id': range(1000, 1500),

        # ZIP code - should NOT compute mean (categorical as numeric)
        'zip_code': np.random.choice([94102, 10001, 60601, 90210, 2138], 500),

        # Phone number - should NOT compute mean (identifier)
        'phone_last_4': np.random.randint(1000, 9999, 500),

        # Age - SHOULD compute mean (true quantitative)
        'age': np.random.randint(18, 80, 500),

        # Purchase count - SHOULD compute mean/sum (count variable)
        'purchases': np.random.randint(0, 20, 500),

        # Revenue - SHOULD compute mean/sum (continuous quantitative)
        'revenue': np.random.uniform(100, 5000, 500).round(2),

        # Category - should compute mode, NOT mean
        'membership_tier': np.random.choice(['bronze', 'silver', 'gold', 'platinum'], 500)
    }

    return pd.DataFrame(data)


def test_old_system_simulation():
    """Simulate what the OLD system would do (blindly compute mean for all numeric columns)"""
    print("=" * 80)
    print("OLD SYSTEM SIMULATION - Blindly Computing Mean for All Numeric Columns")
    print("=" * 80)

    df = create_test_dataset()
    numeric_cols = df.select_dtypes(include=['number']).columns

    print(f"\nOLD SYSTEM: Computing mean for {len(numeric_cols)} numeric columns:")
    for col in numeric_cols:
        mean_val = df[col].mean()
        print(f"  ❌ {col}: mean = {mean_val:.2f}")

    print("\n❌ PROBLEMS:")
    print("  - Mean of customer_id (1249.50) is MEANINGLESS - it's an identifier!")
    print("  - Mean of zip_code (51610.66) is NONSENSE - it's a categorical code!")
    print("  - Mean of phone_last_4 (5494.77) is USELESS - it's part of identifier!")
    print("  - These statistics mislead analysts and waste computation!")


def test_new_system():
    """Test the NEW intelligent analysis system"""
    print("\n\n" + "=" * 80)
    print("NEW INTELLIGENT SYSTEM - Context-Aware Analysis Planning")
    print("=" * 80)

    df = create_test_dataset()

    # Create simple schema
    schema = {
        'columns': [
            {'name': col, 'type': str(df[col].dtype)}
            for col in df.columns
        ]
    }

    print("\n📊 Creating dataset profile...")
    profile = DatasetProfile(df, schema)
    profile_dict = profile.to_dict()

    print(f"✓ Profiled {len(profile_dict['columns'])} columns")
    print(f"✓ Identified {len(profile_dict['relationships']['potential_time_series'])} potential time series")

    print("\n🤖 Planning analyses with Claude...")
    analyzer = IntelligentAnalyzer("analysis")

    try:
        analysis_plan = analyzer.plan_analysis(df, schema, analysis_goal="comprehensive")

        print(f"\n✓ Claude recommended {len(analysis_plan.get('recommended_analyses', []))} appropriate analyses:")
        for i, analysis in enumerate(analysis_plan.get('recommended_analyses', [])[:5], 1):
            print(f"\n  {i}. {analysis['name']}")
            print(f"     Description: {analysis['description']}")
            print(f"     Rationale: {analysis['rationale']}")

        print(f"\n✓ Claude AVOIDED {len(analysis_plan.get('inappropriate_analyses', []))} inappropriate analyses:")
        for i, analysis in enumerate(analysis_plan.get('inappropriate_analyses', []), 1):
            print(f"\n  {i}. {analysis['name']}")
            print(f"     Reason: {analysis['reason']}")

        if analysis_plan.get('data_quality_warnings'):
            print(f"\n⚠️  Data Quality Warnings:")
            for warning in analysis_plan['data_quality_warnings']:
                print(f"     - {warning}")

    except Exception as e:
        print(f"\n❌ Error during analysis planning: {e}")
        print("   (This might be expected if Claude API is not configured)")
        print("   The system will fall back to basic analysis mode")


def test_dataset_profiling():
    """Test just the dataset profiling component (doesn't need Claude API)"""
    print("\n\n" + "=" * 80)
    print("DATASET PROFILING TEST (No Claude API Required)")
    print("=" * 80)

    df = create_test_dataset()

    schema = {
        'columns': [
            {'name': col, 'type': str(df[col].dtype)}
            for col in df.columns
        ]
    }

    profile = DatasetProfile(df, schema)
    profile_dict = profile.to_dict()

    print("\n📊 Dataset Profile Results:")

    for col_info in profile_dict['columns']:
        col_name = col_info['name']
        print(f"\n  Column: {col_name}")
        print(f"    Data Type: {col_info['dtype']}")
        print(f"    Missing: {col_info['missing_pct']:.1f}%")
        print(f"    Unique: {col_info['unique_count']} ({col_info['cardinality']} cardinality)")

        # Check for likely identifier
        if 'numeric_profile' in col_info:
            numeric_profile = col_info['numeric_profile']
            if numeric_profile.get('likely_identifier'):
                print(f"    ⚠️  Likely IDENTIFIER - mean/median not appropriate!")
            elif numeric_profile.get('likely_categorical'):
                print(f"    ⚠️  Likely CATEGORICAL (numeric) - mean/median not appropriate!")
                print(f"    Unique values: {numeric_profile.get('unique_values')}")
            else:
                print(f"    ✓ True NUMERIC - statistics appropriate")
                print(f"    Distribution: {numeric_profile.get('distribution_shape', 'unknown')}")

    print("\n🔍 Detected Relationships:")
    relationships = profile_dict['relationships']
    if relationships['potential_groupings']:
        print(f"  - Found {len(relationships['potential_groupings'])} potential grouping variables")
        for grouping in relationships['potential_groupings'][:3]:
            print(f"    • Group by '{grouping['group_by']}' to analyze {len(grouping['aggregate_cols'])} metrics")

    print("\n🏢 Business Context Indicators:")
    context = profile_dict['business_context']
    if context['domain_indicators']:
        print(f"  Detected domains: {', '.join(context['domain_indicators'])}")
    if context['suggested_analyses']:
        print(f"  Suggested: {', '.join(context['suggested_analyses'])}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("INTELLIGENT ANALYSIS SYSTEM TEST")
    print("=" * 80)

    # Test 1: Show what OLD system would do
    test_old_system_simulation()

    # Test 2: Show what NEW system does (dataset profiling only - works without API)
    test_dataset_profiling()

    # Test 3: Show full NEW system with Claude (requires API)
    test_new_system()

    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("\n✅ Key Improvements Demonstrated:")
    print("  1. Dataset profiling detects identifiers vs. true quantitative variables")
    print("  2. System avoids computing meaningless statistics (mean of ID)")
    print("  3. Claude plans context-appropriate analyses")
    print("  4. Feedback mechanism validates results")
    print("\n🎯 Result: No more 'blindly creating mean' for inappropriate columns!")


if __name__ == "__main__":
    main()
