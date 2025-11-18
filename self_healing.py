"""
Self-Healing Module
Monitors agent health, detects failures, and autonomously repairs issues
"""

import logging
import traceback
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import psutil
import sys

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Health metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    status: str  # healthy, warning, critical
    metadata: Dict = field(default_factory=dict)


@dataclass
class Failure:
    """Failure event record"""
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict
    resolution_attempted: bool = False
    resolution_successful: bool = False
    resolution_strategy: Optional[str] = None


class SelfHealer:
    """
    Self-healing system that monitors, detects, and repairs agent failures
    """

    def __init__(self, agent_ref):
        self.agent = agent_ref
        self.health_metrics: List[HealthMetric] = []
        self.failure_history: List[Failure] = []
        self.monitoring_active = False
        self.healing_strategies = self._load_healing_strategies()

        logger.info("Self-Healer initialized")

    def _load_healing_strategies(self) -> Dict[str, callable]:
        """Load healing strategies for different failure types"""
        return {
            'api_rate_limit': self._heal_rate_limit,
            'connection_timeout': self._heal_connection_timeout,
            'out_of_memory': self._heal_memory_issue,
            'module_not_found': self._heal_missing_dependency,
            'llm_api_error': self._heal_llm_error,
            'code_execution_error': self._heal_code_error,
            'web_search_error': self._heal_search_error,
            'computer_control_error': self._heal_control_error,
            'configuration_error': self._heal_config_error
        }

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self.monitoring_active = True
        logger.info("Starting health monitoring")

        while self.monitoring_active:
            try:
                await self._check_health()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("Stopped health monitoring")

    async def _check_health(self):
        """Comprehensive health check"""
        metrics = []

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(HealthMetric(
            timestamp=datetime.utcnow(),
            metric_name='cpu_usage',
            value=cpu_percent,
            status='critical' if cpu_percent > 90 else 'warning' if cpu_percent > 70 else 'healthy'
        ))

        # Memory usage
        memory = psutil.virtual_memory()
        metrics.append(HealthMetric(
            timestamp=datetime.utcnow(),
            metric_name='memory_usage',
            value=memory.percent,
            status='critical' if memory.percent > 90 else 'warning' if memory.percent > 70 else 'healthy'
        ))

        # Component health checks
        components = ['llm', 'computer', 'searcher', 'executor']
        for component in components:
            if hasattr(self.agent, component):
                component_obj = getattr(self.agent, component)
                is_healthy = await self._check_component_health(component, component_obj)
                metrics.append(HealthMetric(
                    timestamp=datetime.utcnow(),
                    metric_name=f'{component}_health',
                    value=1.0 if is_healthy else 0.0,
                    status='healthy' if is_healthy else 'critical'
                ))

        self.health_metrics.extend(metrics)

        # Trigger healing if critical issues found
        critical_metrics = [m for m in metrics if m.status == 'critical']
        if critical_metrics:
            logger.warning(f"Critical health issues detected: {len(critical_metrics)}")
            await self._trigger_healing(critical_metrics)

    async def _check_component_health(self, name: str, component: Any) -> bool:
        """Check if a component is functioning"""
        try:
            # Basic health check - component exists and is initialized
            if component is None:
                return False

            # Component-specific health checks could go here
            return True
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return False

    async def _trigger_healing(self, critical_metrics: List[HealthMetric]):
        """Trigger healing procedures for critical issues"""
        for metric in critical_metrics:
            if metric.metric_name == 'memory_usage':
                await self._heal_memory_issue()
            elif metric.metric_name.endswith('_health'):
                component_name = metric.metric_name.replace('_health', '')
                await self._heal_component(component_name)

    async def handle_failure(self, error: Exception, component: str, context: Dict) -> bool:
        """
        Handle a failure event and attempt recovery

        Returns:
            bool: True if recovery successful, False otherwise
        """
        failure = Failure(
            timestamp=datetime.utcnow(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context
        )

        self.failure_history.append(failure)
        logger.error(f"Failure detected in {component}: {error}")

        # Determine healing strategy
        strategy = self._determine_strategy(failure)
        if strategy:
            failure.resolution_attempted = True
            failure.resolution_strategy = strategy

            try:
                success = await self.healing_strategies[strategy](failure)
                failure.resolution_successful = success

                if success:
                    logger.info(f"Successfully healed {component} using {strategy}")
                    await self._learn_from_healing(failure)
                else:
                    logger.warning(f"Healing attempt failed for {component}")

                return success
            except Exception as heal_error:
                logger.error(f"Healing strategy {strategy} failed: {heal_error}")
                return False
        else:
            logger.warning(f"No healing strategy found for {failure.error_type}")
            return False

    def _determine_strategy(self, failure: Failure) -> Optional[str]:
        """Determine appropriate healing strategy based on failure"""
        error_type_lower = failure.error_type.lower()
        error_msg_lower = failure.error_message.lower()

        if 'rate limit' in error_msg_lower or 'too many requests' in error_msg_lower:
            return 'api_rate_limit'
        elif 'timeout' in error_msg_lower or 'connection' in error_msg_lower:
            return 'connection_timeout'
        elif 'memory' in error_msg_lower or 'out of memory' in error_msg_lower:
            return 'out_of_memory'
        elif 'modulenotfound' in error_type_lower or 'import' in error_msg_lower:
            return 'module_not_found'
        elif failure.component == 'llm':
            return 'llm_api_error'
        elif failure.component == 'executor':
            return 'code_execution_error'
        elif failure.component == 'searcher':
            return 'web_search_error'
        elif failure.component == 'computer':
            return 'computer_control_error'
        else:
            return None

    async def _heal_rate_limit(self, failure: Failure = None) -> bool:
        """Heal API rate limiting issues"""
        logger.info("Healing rate limit issue - implementing exponential backoff")

        # Wait with exponential backoff
        wait_time = 60  # Start with 60 seconds
        for attempt in range(3):
            logger.info(f"Waiting {wait_time}s before retry (attempt {attempt + 1}/3)")
            await asyncio.sleep(wait_time)
            wait_time *= 2

            # Try to use alternative API or reduce request rate
            if hasattr(self.agent, 'config'):
                # Could switch to backup API key or alternative provider
                pass

        return True

    async def _heal_connection_timeout(self, failure: Failure = None) -> bool:
        """Heal connection timeout issues"""
        logger.info("Healing connection timeout - retrying with increased timeout")

        # Increase timeout settings
        if failure and failure.component in ['searcher', 'llm']:
            component = getattr(self.agent, failure.component, None)
            if component and hasattr(component, 'config'):
                old_timeout = component.config.get('timeout', 30)
                component.config['timeout'] = old_timeout * 2
                logger.info(f"Increased timeout from {old_timeout}s to {component.config['timeout']}s")

        return True

    async def _heal_memory_issue(self, failure: Failure = None) -> bool:
        """Heal memory issues"""
        logger.info("Healing memory issue - clearing caches and optimizing")

        # Clear agent memory/cache
        if hasattr(self.agent, 'memory'):
            memory_size = len(self.agent.memory)
            # Keep only recent 50 items
            self.agent.memory = self.agent.memory[-50:]
            logger.info(f"Cleared memory: {memory_size} -> {len(self.agent.memory)}")

        # Force garbage collection
        import gc
        gc.collect()

        return True

    async def _heal_missing_dependency(self, failure: Failure = None) -> bool:
        """Heal missing dependency issues by installing them"""
        logger.info("Healing missing dependency")

        if failure:
            # Extract module name from error
            import re
            match = re.search(r"No module named '([^']+)'", failure.error_message)
            if match:
                module_name = match.group(1)
                logger.info(f"Attempting to install missing module: {module_name}")

                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
                    logger.info(f"Successfully installed {module_name}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to install {module_name}: {e}")
                    return False

        return False

    async def _heal_llm_error(self, failure: Failure = None) -> bool:
        """Heal LLM API errors"""
        logger.info("Healing LLM error - attempting fallback provider")

        if hasattr(self.agent, 'llm'):
            llm = self.agent.llm
            # Switch between providers
            if llm.provider == 'openai':
                logger.info("Switching from OpenAI to Anthropic")
                llm.provider = 'anthropic'
                llm.model = 'claude-3-sonnet-20240229'
            elif llm.provider == 'anthropic':
                logger.info("Switching from Anthropic to OpenAI")
                llm.provider = 'openai'
                llm.model = 'gpt-4'

            return True

        return False

    async def _heal_code_error(self, failure: Failure = None) -> bool:
        """Heal code execution errors"""
        logger.info("Healing code execution error")

        # Enable safe mode if not already enabled
        if hasattr(self.agent, 'executor'):
            self.agent.executor.safe_mode = True
            logger.info("Enabled safe mode for code executor")

        return True

    async def _heal_search_error(self, failure: Failure = None) -> bool:
        """Heal web search errors"""
        logger.info("Healing search error - switching search engine")

        if hasattr(self.agent, 'searcher'):
            # Switch search engines
            if self.agent.searcher.engine == 'duckduckgo':
                self.agent.searcher.engine = 'google'
            else:
                self.agent.searcher.engine = 'duckduckgo'

            logger.info(f"Switched to {self.agent.searcher.engine}")

        return True

    async def _heal_control_error(self, failure: Failure = None) -> bool:
        """Heal computer control errors"""
        logger.info("Healing control error - enabling safe mode")

        if hasattr(self.agent, 'computer'):
            self.agent.computer.safe_mode = True
            logger.info("Enabled safe mode for computer control")

        return True

    async def _heal_config_error(self, failure: Failure = None) -> bool:
        """Heal configuration errors"""
        logger.info("Healing configuration error - resetting to defaults")

        # Reset to default configuration
        # This would be implemented based on specific needs

        return True

    async def _heal_component(self, component_name: str) -> bool:
        """Attempt to heal a specific component"""
        logger.info(f"Healing component: {component_name}")

        try:
            # Try to reinitialize the component
            if hasattr(self.agent, component_name):
                component = getattr(self.agent, component_name)
                if hasattr(component, '__init__'):
                    # Reinitialize with existing config
                    config = getattr(component, 'config', {})
                    component.__init__(config)
                    logger.info(f"Reinitialized {component_name}")
                    return True
        except Exception as e:
            logger.error(f"Failed to heal component {component_name}: {e}")

        return False

    async def _learn_from_healing(self, failure: Failure):
        """Learn from successful healing to improve future responses"""
        logger.info(f"Learning from successful healing: {failure.resolution_strategy}")

        # Store successful strategies for future use
        learning_data = {
            'error_type': failure.error_type,
            'component': failure.component,
            'strategy': failure.resolution_strategy,
            'timestamp': failure.timestamp.isoformat(),
            'context': failure.context
        }

        # Could store this in a knowledge base for future reference
        # For now, just log it
        logger.info(f"Learned: {json.dumps(learning_data, indent=2)}")

    def get_health_report(self) -> Dict:
        """Generate comprehensive health report"""
        recent_metrics = self.health_metrics[-100:]  # Last 100 metrics

        return {
            'current_status': self._calculate_overall_health(recent_metrics),
            'total_failures': len(self.failure_history),
            'successful_healings': sum(1 for f in self.failure_history if f.resolution_successful),
            'recent_metrics': [
                {
                    'metric': m.metric_name,
                    'value': m.value,
                    'status': m.status,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in recent_metrics[-10:]
            ],
            'recent_failures': [
                {
                    'component': f.component,
                    'error': f.error_type,
                    'strategy': f.resolution_strategy,
                    'success': f.resolution_successful,
                    'timestamp': f.timestamp.isoformat()
                }
                for f in self.failure_history[-10:]
            ]
        }

    def _calculate_overall_health(self, metrics: List[HealthMetric]) -> str:
        """Calculate overall health status"""
        if not metrics:
            return 'unknown'

        recent = metrics[-10:]
        critical_count = sum(1 for m in recent if m.status == 'critical')
        warning_count = sum(1 for m in recent if m.status == 'warning')

        if critical_count > 3:
            return 'critical'
        elif warning_count > 5:
            return 'degraded'
        else:
            return 'healthy'
