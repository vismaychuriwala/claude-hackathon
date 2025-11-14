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
    Example: Run full pipeline on a file using CEO orchestration

    INPUT:
        - file_path: Path to data file

    This demonstrates how to use the system programmatically
    """
    print(f"\n🚀 Running pipeline on: {file_path}\n")

    # Use CEO's run_pipeline method (handles all orchestration)
    results = ceo.run_pipeline(file_path)

    # Display summary
    print("\n" + "=" * 50)
    if results["success"]:
        print("✓ Pipeline completed successfully!")
    else:
        print(f"⚠️  Pipeline completed with errors:")
        for error in results["errors"]:
            print(f"  - {error}")
    print("=" * 50)

    # Display outputs
    if results["data"]:
        print("\nOutputs:")
        print(f"  - Cleaned data: {results['data'].get('cleaned_data_path')}")
        if results["plot"]:
            print(f"  - Plots: {results['plot'].get('total_plots', 0)} files in output/plots/")
        if results["analysis"]:
            print(f"  - Insights: {results['analysis'].get('insights_path')}")
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
