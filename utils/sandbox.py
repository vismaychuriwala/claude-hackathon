"""
Sandbox Execution Utility for Safe Code Execution

Provides a restricted execution environment for agent-generated Python code
with safety controls, timeout handling, and result validation.
"""

import sys
import io
import traceback
import signal
from typing import Dict, Any, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats


class TimeoutException(Exception):
    """Raised when code execution exceeds timeout limit"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Code execution exceeded timeout limit")


class SandboxExecutor:
    """
    Executes Python code in a restricted namespace with safety controls.

    Features:
    - Restricted global namespace (no file I/O, no imports, no system calls)
    - Timeout enforcement
    - Stdout/stderr capture
    - Result validation
    - Error handling with detailed traceback
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize sandbox executor.

        Args:
            timeout: Maximum execution time in seconds (default: 30)
        """
        self.timeout = timeout
        self.allowed_modules = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'scipy_stats': scipy_stats,
            'matplotlib': matplotlib
        }

    def _create_restricted_globals(self, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a restricted global namespace for code execution.

        Args:
            data_context: Dictionary of variables to make available (e.g., {'df': dataframe})

        Returns:
            Dictionary of allowed global variables
        """
        # Restricted builtins (no file operations, imports, etc.)
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'print': print,
            'range': range,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }

        # Combine safe builtins + allowed modules + data context
        restricted_globals = {
            '__builtins__': safe_builtins,
            **self.allowed_modules,
            **data_context
        }

        return restricted_globals

    def execute(
        self,
        code: str,
        data_context: Dict[str, Any],
        return_variable: Optional[str] = None
    ) -> Tuple[bool, Any, str, str]:
        """
        Execute Python code in a restricted sandbox environment.

        Args:
            code: Python code string to execute
            data_context: Variables to make available (e.g., {'df': dataframe})
            return_variable: Optional variable name to return from execution

        Returns:
            Tuple of (success, result, stdout, stderr)
            - success: True if execution succeeded without errors
            - result: Return value (if return_variable specified) or None
            - stdout: Captured standard output
            - stderr: Captured standard error
        """
        # Create restricted namespace
        restricted_globals = self._create_restricted_globals(data_context)
        restricted_locals = {}

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        success = False
        result = None

        try:
            # Set timeout alarm (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)

            # Execute code with output capture
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals, restricted_locals)

            # Cancel timeout alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

            # Get return value if specified
            if return_variable and return_variable in restricted_locals:
                result = restricted_locals[return_variable]
            elif return_variable and return_variable in restricted_globals:
                result = restricted_globals[return_variable]

            success = True

        except TimeoutException as e:
            stderr_capture.write(f"TIMEOUT ERROR: {str(e)}\n")
            success = False

        except Exception as e:
            # Capture full traceback
            tb = traceback.format_exc()
            stderr_capture.write(f"EXECUTION ERROR:\n{tb}\n")
            success = False

        finally:
            # Ensure timeout alarm is cancelled
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        return success, result, stdout_text, stderr_text

    def validate_result(
        self,
        result: Any,
        expected_type: Optional[type] = None,
        validation_func: Optional[callable] = None
    ) -> Tuple[bool, str]:
        """
        Validate execution result.

        Args:
            result: Result from code execution
            expected_type: Expected type of result (optional)
            validation_func: Custom validation function (optional)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if result exists
        if result is None:
            return False, "No result returned from execution"

        # Check expected type
        if expected_type and not isinstance(result, expected_type):
            return False, f"Expected type {expected_type.__name__}, got {type(result).__name__}"

        # Custom validation
        if validation_func:
            try:
                is_valid = validation_func(result)
                if not is_valid:
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {str(e)}"

        return True, ""


class CodeValidationResult:
    """Result of code validation and execution"""

    def __init__(
        self,
        success: bool,
        result: Any = None,
        stdout: str = "",
        stderr: str = "",
        validation_error: str = "",
        execution_time: float = 0.0
    ):
        self.success = success
        self.result = result
        self.stdout = stdout
        self.stderr = stderr
        self.validation_error = validation_error
        self.execution_time = execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'success': self.success,
            'result': self._serialize_result(self.result),
            'stdout': self.stdout,
            'stderr': self.stderr,
            'validation_error': self.validation_error,
            'execution_time': self.execution_time
        }

    def _serialize_result(self, result: Any) -> Any:
        """Convert result to JSON-serializable format"""
        if result is None:
            return None
        elif isinstance(result, (str, int, float, bool)):
            return result
        elif isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}
        elif isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result]
        elif isinstance(result, pd.DataFrame):
            return result.to_dict(orient='records')
        elif isinstance(result, pd.Series):
            return result.to_dict()
        elif isinstance(result, np.ndarray):
            return result.tolist()
        else:
            return str(result)


def execute_analysis_code(
    code: str,
    df: pd.DataFrame,
    return_variable: str = "analysis_result",
    timeout: int = 30
) -> CodeValidationResult:
    """
    Convenience function to execute analysis code on a DataFrame.

    Args:
        code: Python code to execute
        df: DataFrame to analyze
        return_variable: Variable name containing the result
        timeout: Timeout in seconds

    Returns:
        CodeValidationResult with execution details
    """
    import time

    executor = SandboxExecutor(timeout=timeout)

    start_time = time.time()
    success, result, stdout, stderr = executor.execute(
        code=code,
        data_context={'df': df},
        return_variable=return_variable
    )
    execution_time = time.time() - start_time

    # Validate result
    validation_error = ""
    if success and result is None:
        validation_error = f"Variable '{return_variable}' was not found in execution scope"
        success = False

    return CodeValidationResult(
        success=success,
        result=result,
        stdout=stdout,
        stderr=stderr,
        validation_error=validation_error,
        execution_time=execution_time
    )
