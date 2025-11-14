"""
Main Entry Point - Multi-Agent Data Command Center
Sets up CEO and registers all agents
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ceo import ceo, AgentRequest
from agents import DataAgent, PlotAgent, AnalysisAgent


def setup_agents():
    """
    Register all agents with CEO

    This must be called before running the pipeline
    """
    print("=" * 50)
    print("Multi-Agent Data Command Center")
    print("=" * 50)

    # Create agent instances
    data_agent = DataAgent()
    plot_agent = PlotAgent()
    analysis_agent = AnalysisAgent()

    # Register with CEO
    ceo.register_agent("data", data_agent)
    ceo.register_agent("plot", plot_agent)
    ceo.register_agent("analysis", analysis_agent)

    print("\n✓ All agents registered with CEO")
    print("\nTeam Assignments:")
    print("  - Vismay: Data Pipeline Agent")
    print("  - Nikunj: Visualization Agent")
    print("  - Shamanth: Analysis Agent")
    print("  - Amit: UI Developer")
    print("=" * 50)


def run_pipeline_example(file_path: str):
    """
    Example: Run full pipeline on a file

    INPUT:
        - file_path: Path to data file

    This demonstrates how to use the system programmatically
    """
    print(f"\n🚀 Running pipeline on: {file_path}")

    # Step 1: Data processing
    print("\n[1/3] Data Agent processing...")
    data_request = AgentRequest("data", "process_file", {"file_path": file_path})
    data_response = ceo.execute_with_retry(data_request)

    if not data_response.success:
        print(f"❌ Data processing failed: {data_response.error}")
        return

    print(f"✓ Data processed: {data_response.data.get('cleaned_data_path')}")

    # Step 2: Plotting
    print("\n[2/3] Plot Agent creating visualizations...")
    plot_request = AgentRequest("plot", "create_plots", data_response.data)
    plot_response = ceo.execute_with_retry(plot_request)

    if not plot_response.success:
        print(f"❌ Plotting failed: {plot_response.error}")
        return

    print(f"✓ Created {plot_response.data.get('total_plots', 0)} plots")

    # Step 3: Analysis
    print("\n[3/3] Analysis Agent generating insights...")
    analysis_request = AgentRequest("analysis", "generate_insights", {})
    analysis_response = ceo.execute_with_retry(analysis_request)

    if not analysis_response.success:
        print(f"❌ Analysis failed: {analysis_response.error}")
        return

    print(f"✓ Generated {analysis_response.data.get('insights_count', 0)} insights")

    print("\n" + "=" * 50)
    print("✓ Pipeline complete!")
    print("=" * 50)
    print("\nOutputs:")
    print(f"  - Cleaned data: {data_response.data.get('cleaned_data_path')}")
    print(f"  - Plots: {plot_response.data.get('total_plots')} files in output/plots/")
    print(f"  - Insights: {analysis_response.data.get('insights_path')}")
    print(f"  - Reports: output/reports/")


def start_ui():
    """
    Start the web UI (Flask app)
    """
    from ui.app import app
    print("\n🌐 Starting web UI on http://localhost:5000")
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    # Setup agents
    setup_agents()

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "ui":
            # Start web UI
            start_ui()
        else:
            # Run pipeline on provided file
            file_path = sys.argv[1]
            if Path(file_path).exists():
                run_pipeline_example(file_path)
            else:
                print(f"❌ File not found: {file_path}")
    else:
        # Default: Show usage
        print("\nUsage:")
        print("  python main.py ui                 # Start web UI")
        print("  python main.py <file_path>       # Run pipeline on file")
        print("\nExample:")
        print("  python main.py ui")
        print("  python main.py data/sample.csv")
