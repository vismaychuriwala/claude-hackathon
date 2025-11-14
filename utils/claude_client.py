"""
Claude API Client Wrapper
Handles all interactions with Claude API
"""
import anthropic
from typing import List, Dict, Optional
from config.config import CLAUDE_API_KEY, CLAUDE_MODEL


class ClaudeClient:
    """Wrapper for Claude API calls"""

    def __init__(self, api_key: str = CLAUDE_API_KEY):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = CLAUDE_MODEL

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Make a call to Claude API

        INPUT:
            - prompt: The user prompt/task
            - system_prompt: Optional system instructions
            - max_tokens: Maximum tokens in response

        OUTPUT:
            - str: Claude's response text
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def call_with_context(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096
    ) -> str:
        """
        Make a call with conversation context

        INPUT:
            - messages: List of {"role": "user/assistant", "content": "..."}
            - system_prompt: Optional system instructions
            - max_tokens: Maximum tokens in response

        OUTPUT:
            - str: Claude's response text
        """
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


# Global client instance
claude = ClaudeClient()
