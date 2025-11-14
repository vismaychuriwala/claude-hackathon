"""
CEO Orchestrator - Central coordinator for all agent actions
Routes requests, handles retries, manages workflow
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from config.config import STATUS_FILE, MAX_RETRIES


class AgentRequest:
    """Represents a request to an agent"""

    def __init__(self, agent_name: str, action: str, data: Dict[str, Any]):
        self.agent_name = agent_name
        self.action = action
        self.data = data
        self.retry_count = 0
        self.status = "pending"  # pending, in_progress, success, failure
        self.result = None
        self.error_message = None


class AgentResponse:
    """Represents a response from an agent"""

    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        retry_requested: bool = False,
        retry_reason: Optional[str] = None
    ):
        self.success = success
        self.data = data or {}
        self.error = error
        self.retry_requested = retry_requested
        self.retry_reason = retry_reason


class CEOOrchestrator:
    """
    Central coordinator that routes all agent actions
    Handles retries, manages workflow, logs status
    """

    def __init__(self):
        self.agents = {}  # Registry of available agents
        self.request_queue = []  # Queue of pending requests
        self.status = self._load_status()

    def register_agent(self, name: str, agent_instance):
        """
        Register an agent with the CEO

        INPUT:
            - name: Agent name (e.g., "data", "plot", "analysis")
            - agent_instance: The agent object with execute() method
        """
        self.agents[name] = agent_instance
        print(f"[CEO] Registered agent: {name}")

    def route_request(self, request: AgentRequest) -> AgentResponse:
        """
        Route a request to the appropriate agent

        INPUT:
            - request: AgentRequest object

        OUTPUT:
            - AgentResponse object
        """
        agent = self.agents.get(request.agent_name)
        if not agent:
            return AgentResponse(
                success=False,
                error=f"Agent '{request.agent_name}' not found"
            )

        # Update status
        self._update_status(request.agent_name, "in_progress", f"Executing {request.action}")
        request.status = "in_progress"

        try:
            # Execute the agent's action
            result = agent.execute(request.action, request.data)
            request.status = "success"
            request.result = result

            self._update_status(request.agent_name, "completed", f"Completed {request.action}")

            return AgentResponse(success=True, data=result)

        except Exception as e:
            request.status = "failure"
            request.error_message = str(e)

            self._update_status(
                request.agent_name,
                "failed",
                f"Error: {str(e)}"
            )

            return AgentResponse(success=False, error=str(e))

    def execute_with_retry(self, request: AgentRequest) -> AgentResponse:
        """
        Execute request with automatic retry on failure

        INPUT:
            - request: AgentRequest object

        OUTPUT:
            - AgentResponse object (final result after retries)
        """
        response = None

        while request.retry_count < MAX_RETRIES:
            response = self.route_request(request)

            if response.success:
                return response

            # Check if retry requested by agent
            if response.retry_requested and request.retry_count < MAX_RETRIES - 1:
                request.retry_count += 1
                print(f"[CEO] Retry {request.retry_count}/{MAX_RETRIES} for {request.agent_name}: {response.retry_reason}")
                self._update_status(
                    request.agent_name,
                    "retrying",
                    f"Retry {request.retry_count}: {response.retry_reason}"
                )
                continue
            else:
                break

        return response

    def run_pipeline(self, file_path: str) -> Dict[str, Any]:
        """
        Run the full pipeline: Data -> Plotting -> Analysis

        INPUT:
            - file_path: Path to uploaded file

        OUTPUT:
            - Dict with results from each stage
        """
        print(f"[CEO] Starting pipeline for: {file_path}")
        results = {
            "data": None,
            "plot": None,
            "analysis": None,
            "success": True,
            "errors": []
        }

        # Stage 1: Data Processing
        print("[CEO] Stage 1/3: Data Processing")
        data_request = AgentRequest("data", "process_file", {"file_path": file_path})
        data_response = self.execute_with_retry(data_request)

        if not data_response.success:
            error_msg = f"Data processing failed: {data_response.error}"
            print(f"[CEO] ❌ {error_msg}")
            results["success"] = False
            results["errors"].append(error_msg)
            return results

        results["data"] = data_response.data
        print(f"[CEO] ✓ Data processing complete")

        # Stage 2: Plot Generation
        print("[CEO] Stage 2/3: Plot Generation")
        plot_request = AgentRequest("plot", "create_plots", {
            "cleaned_data_path": data_response.data.get("cleaned_data_path"),
            "schema_path": data_response.data.get("schema_path")
        })
        plot_response = self.execute_with_retry(plot_request)

        if not plot_response.success:
            error_msg = f"Plot generation failed: {plot_response.error}"
            print(f"[CEO] ⚠️  {error_msg}")
            results["errors"].append(error_msg)
            # Continue to analysis even if plotting fails
        else:
            results["plot"] = plot_response.data
            print(f"[CEO] ✓ Plot generation complete")

        # Stage 3: Analysis
        print("[CEO] Stage 3/3: Analysis")
        analysis_request = AgentRequest("analysis", "generate_insights", {
            "cleaned_data_path": data_response.data.get("cleaned_data_path"),
            "schema_path": data_response.data.get("schema_path"),
            "plot_metadata_path": plot_response.data.get("plot_metadata_path") if plot_response.success else None,
            "transformation_log_path": data_response.data.get("transformation_log_path")
        })
        analysis_response = self.execute_with_retry(analysis_request)

        if not analysis_response.success:
            error_msg = f"Analysis failed: {analysis_response.error}"
            print(f"[CEO] ⚠️  {error_msg}")
            results["errors"].append(error_msg)
        else:
            results["analysis"] = analysis_response.data
            print(f"[CEO] ✓ Analysis complete")

        # Pipeline complete
        if len(results["errors"]) > 0:
            results["success"] = False
            print(f"[CEO] ⚠️  Pipeline completed with {len(results['errors'])} errors")
        else:
            print("[CEO] ✓ Pipeline completed successfully!")

        return results

    def _update_status(self, stage: str, status: str, message: str):
        """
        Update status.json with current progress

        INPUT:
            - stage: Agent/stage name
            - status: Status (pending, in_progress, completed, failed, retrying)
            - message: Human-readable message
        """
        self.status["stages"][stage] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }

        # Determine current stage
        if status == "in_progress":
            self.status["current_stage"] = stage

        # Save to file
        self._save_status()

    def _load_status(self) -> Dict[str, Any]:
        """Load status from status.json or create new"""
        if STATUS_FILE.exists():
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        else:
            return {
                "current_stage": None,
                "stages": {},
                "errors": [],
                "retry_count": 0
            }

    def _save_status(self):
        """Save status to status.json"""
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w") as f:
            json.dump(self.status, f, indent=2)


# Global CEO instance
ceo = CEOOrchestrator()
