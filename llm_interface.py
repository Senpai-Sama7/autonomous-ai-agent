"""
LLM Interface Module
Handles communication with language models for planning and reasoning
"""

import logging
from typing import Dict, List, Any, Optional
import anthropic
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMInterface:
    """Handles LLM API calls for planning and reasoning"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM interface

        Args:
            config: Configuration with API keys and model settings
        """
        self.config = config
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)

        # Initialize appropriate client
        if 'claude' in self.model.lower():
            self.client = Anthropic(api_key=config.get('anthropic_api_key'))
            self.provider = 'anthropic'
        else:
            openai.api_key = config.get('openai_api_key')
            self.provider = 'openai'

        logger.info(f"LLM Interface initialized with model: {self.model}")

    async def query(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a query to the LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            LLM response text
        """
        try:
            if self.provider == 'anthropic':
                return await self._query_anthropic(prompt, system_prompt)
            else:
                return await self._query_openai(prompt, system_prompt)
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return f"Error: {str(e)}"

    async def _query_anthropic(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Query Anthropic Claude"""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    async def _query_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Query OpenAI GPT"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    async def create_plan(self, task_description: str, memory: List[Dict]) -> Dict:
        """
        Create an execution plan for a task

        Args:
            task_description: Description of the task
            memory: Agent's memory of previous actions

        Returns:
            Execution plan dictionary
        """
        system_prompt = """You are an AI agent planner. Given a task description and memory of previous actions,
        create a detailed execution plan. Return a JSON object with the following structure:
        {
            "steps": [
                {
                    "action": "action_type",
                    "parameters": {...},
                    "description": "what this step does"
                }
            ],
            "reasoning": "why this plan will accomplish the task"
        }

        Available actions:
        - search_web: Search the web for information
        - execute_code: Execute code in a safe environment
        - control_computer: Control mouse, keyboard, or take screenshots
        - llm_query: Ask the LLM a question
        """

        memory_summary = self._summarize_memory(memory)

        prompt = f"""Task: {task_description}

Previous actions:
{memory_summary}

Create an execution plan for this task."""

        response = await self.query(prompt, system_prompt)

        try:
            # Parse JSON response
            import json
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            # Fallback plan
            return {
                "steps": [
                    {
                        "action": "llm_query",
                        "parameters": {"prompt": task_description},
                        "description": "Process task with LLM"
                    }
                ],
                "reasoning": "Fallback plan due to parsing error"
            }

    async def determine_next_action(self, goal: str, memory: List[Dict]) -> Dict:
        """
        Determine the next action in autonomous mode

        Args:
            goal: Overall goal to achieve
            memory: Agent's memory

        Returns:
            Next action dictionary
        """
        system_prompt = """You are an autonomous AI agent. Analyze the goal and previous actions,
        then determine the next best action to take. Return JSON:
        {
            "action": "action_type or 'goal_achieved'",
            "parameters": {...},
            "description": "what this action will do",
            "reasoning": "why this is the best next step"
        }"""

        memory_summary = self._summarize_memory(memory)

        prompt = f"""Goal: {goal}

Actions taken so far:
{memory_summary}

What should be the next action?"""

        response = await self.query(prompt, system_prompt)

        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "action": "goal_achieved",
                "description": "Unable to determine next action",
                "reasoning": "JSON parsing error"
            }

    def _summarize_memory(self, memory: List[Dict], max_items: int = 10) -> str:
        """Summarize memory for prompts"""
        if not memory:
            return "None"

        recent_memory = memory[-max_items:]
        summary_lines = []

        for i, item in enumerate(recent_memory, 1):
            step = item.get('step', {})
            result = item.get('result', {})
            summary_lines.append(
                f"{i}. {step.get('action', 'unknown')}: {step.get('description', 'N/A')}"
            )

        return "\n".join(summary_lines)
