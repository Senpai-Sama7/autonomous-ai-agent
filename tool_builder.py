"""
Tool Builder Module
Dynamically creates, tests, and integrates new tools and capabilities
"""

import logging
import ast
import inspect
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for a new tool"""
    name: str
    description: str
    input_schema: Dict
    output_schema: Dict
    implementation_requirements: List[str]


class ToolBuilder:
    """
    Autonomously builds and integrates new tools
    """

    def __init__(self, agent_ref):
        self.agent = agent_ref
        self.built_tools: Dict[str, Callable] = {}
        self.tool_registry: Dict[str, ToolSpec] = {}

        logger.info("Tool Builder initialized")

    async def identify_needed_tool(self, task: str, current_capabilities: List[str]) -> Optional[ToolSpec]:
        """Identify if a new tool is needed"""
        prompt = f"""Analyze if a new tool is needed for this task:

        Task: {task}
        Current capabilities: {current_capabilities}

        If a new tool would significantly help, provide:
        1. Tool name
        2. Description
        3. Input parameters
        4. Output format
        5. Implementation requirements

        If existing capabilities are sufficient, respond with "NOT_NEEDED"."""

        response = await self.agent.llm.query(prompt)

        if "NOT_NEEDED" in response:
            return None

        # Parse response to create ToolSpec
        # (Simplified - in reality would use structured output)
        tool_spec = self._parse_tool_spec(response)
        return tool_spec

    def _parse_tool_spec(self, llm_response: str) -> ToolSpec:
        """Parse LLM response into ToolSpec"""
        # Simplified parsing
        lines = llm_response.strip().split('\n')

        return ToolSpec(
            name=lines[0] if lines else "new_tool",
            description=lines[1] if len(lines) > 1 else "",
            input_schema={},
            output_schema={},
            implementation_requirements=lines[2:] if len(lines) > 2 else []
        )

    async def build_tool(self, spec: ToolSpec) -> Optional[Callable]:
        """Build a new tool from specification"""
        logger.info(f"Building tool: {spec.name}")

        # Generate implementation code
        code = await self._generate_tool_code(spec)

        # Test the tool
        if await self._test_tool_code(code):
            # Integrate the tool
            tool_func = await self._compile_and_integrate(code, spec.name)

            if tool_func:
                self.built_tools[spec.name] = tool_func
                self.tool_registry[spec.name] = spec
                logger.info(f"Successfully built and integrated tool: {spec.name}")
                return tool_func

        logger.warning(f"Failed to build tool: {spec.name}")
        return None

    async def _generate_tool_code(self, spec: ToolSpec) -> str:
        """Generate Python code for the tool"""
        prompt = f"""Generate complete, working Python code for this tool:

        Name: {spec.name}
        Description: {spec.description}
        Input schema: {spec.input_schema}
        Output schema: {spec.output_schema}
        Requirements: {spec.implementation_requirements}

        Provide a complete async function with:
        - Proper error handling
        - Input validation
        - Clear documentation
        - Type hints

        Code only, no explanations:"""

        code = await self.agent.llm.query(prompt)
        return code

    async def _test_tool_code(self, code: str) -> bool:
        """Test if generated code is valid and safe"""
        logger.info("Testing generated tool code")

        try:
            # Parse to check syntax
            ast.parse(code)

            # Check for dangerous operations
            dangerous_patterns = ['os.system', '__import__', 'eval', 'exec']
            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    logger.warning(f"Dangerous pattern detected: {pattern}")
                    return False

            # Try to execute in safe environment
            result = await self.agent.executor.execute(code, 'python')

            return result['success']

        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False

    async def _compile_and_integrate(self, code: str, tool_name: str) -> Optional[Callable]:
        """Compile code and create callable tool"""
        try:
            # Create namespace for execution
            namespace = {}
            exec(code, namespace)

            # Find the main function (assumed to be named after tool or first async def)
            tool_func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    tool_func = obj
                    break

            if tool_func:
                logger.info(f"Compiled tool: {tool_name}")
                return tool_func

            return None

        except Exception as e:
            logger.error(f"Failed to compile tool: {e}")
            return None

    async def auto_improve_tool(self, tool_name: str, usage_feedback: List[Dict]):
        """Automatically improve an existing tool based on usage"""
        if tool_name not in self.built_tools:
            logger.warning(f"Tool {tool_name} not found")
            return

        logger.info(f"Auto-improving tool: {tool_name}")

        # Analyze feedback
        improvements = await self._analyze_feedback(usage_feedback)

        # Regenerate with improvements
        spec = self.tool_registry[tool_name]
        spec.implementation_requirements.extend(improvements)

        # Rebuild tool
        await self.build_tool(spec)

    async def _analyze_feedback(self, feedback: List[Dict]) -> List[str]:
        """Analyze usage feedback to identify improvements"""
        prompt = f"""Analyze this tool usage feedback and suggest improvements:

        Feedback: {json.dumps(feedback, indent=2)}

        Provide 3-5 specific improvement suggestions:"""

        response = await self.agent.llm.query(prompt)
        improvements = [line.strip() for line in response.split('\n') if line.strip()]

        return improvements
