"""
Claude API Client Wrapper
Handles all interactions with Claude API with support for:
- Configurable models per agent
- Extended thinking (reasoning) mode
- Conversation context
"""
import anthropic
from typing import List, Dict, Optional, Any
from config.config import CLAUDE_API_KEY, CLAUDE_MODEL, AGENT_MODELS, EXTENDED_THINKING


class ClaudeClient:
    """Wrapper for Claude API calls with extended thinking support"""

    def __init__(self, api_key: str = CLAUDE_API_KEY, agent_name: Optional[str] = None):
        """
        Initialize Claude client

        Args:
            api_key: Anthropic API key
            agent_name: Name of agent (data/plot/analysis) for model selection
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.agent_name = agent_name
        # Use agent-specific model if available, otherwise default
        self.model = AGENT_MODELS.get(agent_name, CLAUDE_MODEL) if agent_name else CLAUDE_MODEL

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Make a call to Claude API

        Args:
            prompt: The user prompt/task
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            model: Override model (optional)
            thinking: Extended thinking config, e.g., {"type": "enabled", "budget_tokens": 5000}

        Returns:
            Claude's response text (excludes thinking blocks)
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        # Add extended thinking if configured
        if thinking:
            kwargs["thinking"] = thinking

        response = self.client.messages.create(**kwargs)

        # Extract text content only (skip thinking blocks)
        text_content = []
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'text':
                text_content.append(block.text)

        return '\n'.join(text_content) if text_content else response.content[0].text

    def call_with_context(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        thinking: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Make a call with conversation context

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            model: Override model (optional)
            thinking: Extended thinking config

        Returns:
            Claude's response text
        """
        kwargs = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if thinking:
            kwargs["thinking"] = thinking

        response = self.client.messages.create(**kwargs)

        # Extract text content only
        text_content = []
        for block in response.content:
            if hasattr(block, 'type') and block.type == 'text':
                text_content.append(block.text)

        return '\n'.join(text_content) if text_content else response.content[0].text

    def get_thinking_config(self, enabled: bool = None, budget_tokens: int = None) -> Dict[str, Any]:
        """
        Helper to create thinking configuration

        Args:
            enabled: Override global thinking setting
            budget_tokens: Token budget (min 1024)

        Returns:
            Thinking config dict or None
        """
        is_enabled = enabled if enabled is not None else EXTENDED_THINKING.get("enabled", False)

        if not is_enabled:
            return None

        budget = budget_tokens or EXTENDED_THINKING.get("budget_tokens", 5000)
        return {
            "type": "enabled",
            "budget_tokens": max(1024, budget)  # Ensure minimum
        }


# Global client instance (default, no agent-specific model)
claude = ClaudeClient()
