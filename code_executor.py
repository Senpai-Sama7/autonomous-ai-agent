"""
Code Executor Module
Safely executes code in isolated environments
"""

import subprocess
import tempfile
import os
import logging
from typing import Dict, Any, Optional
import docker
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeExecutor:
    """Handles safe code execution in isolated environments"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize code executor

        Args:
            config: Configuration with safety settings
        """
        self.config = config
        self.safe_mode = config.get('safe_mode', True)
        self.timeout = config.get('timeout', 30)
        self.use_docker = config.get('use_docker', False)

        if self.use_docker:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Docker not available: {e}")
                self.use_docker = False

        logger.info(f"Code Executor initialized (safe_mode={self.safe_mode})")

    async def execute(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Execute code in specified language

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash, etc.)

        Returns:
            Execution result dictionary
        """
        logger.info(f"Executing {language} code ({len(code)} chars)")

        if self.safe_mode and not self._is_code_safe(code, language):
            return {
                'success': False,
                'error': 'Code safety check failed',
                'output': ''
            }

        if self.use_docker:
            return await self._execute_in_docker(code, language)
        else:
            return await self._execute_local(code, language)

    def _is_code_safe(self, code: str, language: str) -> bool:
        """
        Check if code is safe to execute

        Args:
            code: Code to check
            language: Programming language

        Returns:
            True if code appears safe
        """
        # List of potentially dangerous operations
        dangerous_keywords = [
            'os.system', 'subprocess.call', 'eval(', 'exec(',
            '__import__', 'open(', 'file(', 'rm -rf', 'dd if=',
            'format(', 'mkfs', 'fdisk'
        ]

        code_lower = code.lower()

        for keyword in dangerous_keywords:
            if keyword.lower() in code_lower:
                logger.warning(f"Potentially dangerous keyword detected: {keyword}")
                return False

        return True

    async def _execute_local(self, code: str, language: str) -> Dict[str, Any]:
        """Execute code locally in a subprocess"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=self._get_file_extension(language),
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Get command for language
                command = self._get_execution_command(language, temp_file)

                # Execute
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                return {
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr,
                    'return_code': result.returncode
                }

            finally:
                # Clean up temp file
                os.unlink(temp_file)

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Execution timeout ({self.timeout}s)',
                'output': ''
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output': ''
            }

    async def _execute_in_docker(self, code: str, language: str) -> Dict[str, Any]:
        """Execute code in Docker container"""
        try:
            # Get appropriate Docker image
            image = self._get_docker_image(language)

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=self._get_file_extension(language),
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Run in container
                container = self.docker_client.containers.run(
                    image,
                    command=self._get_docker_command(language, os.path.basename(temp_file)),
                    volumes={
                        os.path.dirname(temp_file): {
                            'bind': '/code',
                            'mode': 'ro'
                        }
                    },
                    working_dir='/code',
                    detach=False,
                    remove=True,
                    mem_limit='512m',
                    network_disabled=True,
                    timeout=self.timeout
                )

                output = container.decode('utf-8')

                return {
                    'success': True,
                    'output': output,
                    'error': '',
                    'execution': 'docker'
                }

            finally:
                os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'output': '',
                'execution': 'docker'
            }

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'bash': '.sh',
            'ruby': '.rb',
            'go': '.go'
        }
        return extensions.get(language, '.txt')

    def _get_execution_command(self, language: str, filepath: str) -> str:
        """Get execution command for language"""
        commands = {
            'python': f'python3 {filepath}',
            'javascript': f'node {filepath}',
            'bash': f'bash {filepath}',
            'ruby': f'ruby {filepath}',
            'go': f'go run {filepath}'
        }
        return commands.get(language, f'cat {filepath}')

    def _get_docker_image(self, language: str) -> str:
        """Get Docker image for language"""
        images = {
            'python': 'python:3.11-slim',
            'javascript': 'node:18-slim',
            'bash': 'bash:latest',
            'ruby': 'ruby:3.2-slim',
            'go': 'golang:1.21-alpine'
        }
        return images.get(language, 'python:3.11-slim')

    def _get_docker_command(self, language: str, filename: str) -> str:
        """Get Docker execution command"""
        commands = {
            'python': f'python {filename}',
            'javascript': f'node {filename}',
            'bash': f'bash {filename}',
            'ruby': f'ruby {filename}',
            'go': f'go run {filename}'
        }
        return commands.get(language, f'cat {filename}')
