import asyncio
import logging
from typing import Dict, Any
from bt_agent.core import (
    BTAgent, BTAgentAction, BTAgentAsyncAction, BTToolAction, BTHandoffAction, 
    BTConditionNode, BTExecutionContext, create_retry_decorator, create_timeout_decorator
)

# Try to import from new BTEngine first, fall back to compatibility
try:
    from behavior_tree_engine.core import NodeStatus, Node as BTNode
    from behavior_tree_engine.core import Sequence as SequenceNode, Selector as SelectorNode, Parallel as ParallelNode
    from behavior_tree_engine.core import Timeout, AsyncAction, BehaviorTree
    NEW_BTENGINE = True
    print("✅ Using NEW BTEngine with all features!")
except ImportError:
    try:
        # Fall back to old imports
        from behavior_tree_engine.core import NodeStatus, Node as BTNode
        from behavior_tree_engine.core import Sequence as SequenceNode, Selector as SelectorNode, Parallel as ParallelNode
        from behavior_tree_engine.core import MaxTSec as Timeout
        NEW_BTENGINE = False
        print("⚠️  Using OLD BTEngine - some features may be limited")
    except ImportError as e:
        print(f"❌ BTEngine not found: {e}")
        print("🔧 Please run: python reset_btengine.py")
        raise

from agents import function_tool
from agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example tools for complex task execution (same as before)
async def _analyze_requirements(task_description: str) -> str:
    """Analyze task requirements and break down into subtasks."""
    logger.info(f"Analyzing requirements for: {task_description}")
    
    # Simulate requirement analysis
    subtasks = [
        {"id": 1, "name": "data_collection", "priority": "high"},
        {"id": 2, "name": "data_processing", "priority": "medium"},
        {"id": 3, "name": "result_validation", "priority": "high"}
    ]
    
    result = {
        "subtasks": subtasks,
        "complexity": "medium",
        "estimated_time": "30 minutes"
    }
    
    return str(result)

async def _collect_data(data_source: str, query_params: str = "{}") -> str:
    """Collect data from specified source."""
    logger.info(f"Collecting data from {data_source} with params: {query_params}")
    
    # Simulate data collection
    result = {
        "status": "success",
        "records_collected": 150,
        "data_quality": "good",
        "source": data_source
    }
    
    return str(result)

async def _process_data(data: str, processing_type: str = "standard") -> str:
    """Process collected data."""
    logger.info(f"Processing data with type: {processing_type}")
    
    # Simulate data processing
    result = {
        "status": "completed",
        "processed_records": 150,
        "processing_time": "5 minutes",
        "quality_score": 0.85
    }
    
    return str(result)

async def _validate_results(results: str) -> str:
    """Validate processing results."""
    logger.info("Validating processing results")
    
    # Simple validation logic
    quality_score = 0.85
    is_valid = quality_score > 0.8
    
    result = {
        "is_valid": is_valid,
        "quality_score": quality_score,
        "validation_status": "passed" if is_valid else "failed"
    }
    
    return str(result)

# Create function tools from the underlying functions
analyze_requirements = function_tool(_analyze_requirements)
collect_data = function_tool(_collect_data)
process_data = function_tool(_process_data)
validate_results = function_tool(_validate_results)

# Compatible async action base class
if NEW_BTENGINE:
    # Use the new AsyncAction base class
    class CompatibleAsyncAction(BTAgentAsyncAction):
        """Using new AsyncAction base class."""
        pass
else:
    # Use the old action base class with async compatibility
    class CompatibleAsyncAction(BTAgentAction):
        """Compatibility wrapper for old BTEngine."""
        
        async def execute_async(self) -> NodeStatus:
            """Override this method for async execution logic."""
            return NodeStatus.SUCCESS
        
        def _tick(self) -> NodeStatus:
            """Execute with async compatibility."""
            try:
                # For old BTEngine, run async code synchronously
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already in an async context, create a task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, self.execute_async())
                            return future.result(timeout=30)
                    else:
                        return loop.run_until_complete(self.execute_async())
                except RuntimeError:
                    # No event loop, create one
                    return asyncio.run(self.execute_async())
            except Exception as e:
                logger.error(f"Compatible async action {self.name} failed: {e}")
                return NodeStatus.FAILURE

# Custom action nodes for complex task execution
class RequirementAnalysisAction(CompatibleAsyncAction):
    """Custom async action node for analyzing task requirements."""
    
    async def execute_async(self) -> NodeStatus:
        """Execute requirements analysis."""
        try:
            task_description = self.get_shared_data("task_description", "Default task")
            result_str = await _analyze_requirements(task_description)
            
            # Store results in shared memory
            self.set_shared_data("requirements_analysis", result_str)
            self.set_shared_data("subtasks", "analysis_completed")
            
            logger.info(f"Requirements analysis completed: {result_str}")
            return NodeStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return NodeStatus.FAILURE

class DataCollectionAction(CompatibleAsyncAction):
    """Custom async action node for data collection."""
    
    async def execute_async(self) -> NodeStatus:
        """Execute data collection."""
        try:
            data_source = self.get_shared_data("data_source", "default_database")
            query_params = str(self.get_shared_data("query_params", {}))
            result_str = await _collect_data(data_source, query_params)
            
            # Store results
            self.set_shared_data("collected_data", result_str)
            
            # Check if collection was successful
            if "success" in result_str.lower():
                logger.info(f"Data collection successful: {result_str}")
                return NodeStatus.SUCCESS
            else:
                logger.warning("Data collection failed")
                return NodeStatus.FAILURE
                
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return NodeStatus.FAILURE

class DataProcessingAction(CompatibleAsyncAction):
    """Custom async action node for data processing."""
    
    async def execute_async(self) -> NodeStatus:
        """Execute data processing."""
        try:
            collected_data = self.get_shared_data("collected_data")
            if not collected_data:
                logger.error("No collected data available for processing")
                return NodeStatus.FAILURE
            
            processing_type = self.get_shared_data("processing_type", "standard")
            result_str = await _process_data(str(collected_data), processing_type)
            
            # Store results
            self.set_shared_data("processed_data", result_str)
            
            logger.info(f"Data processing completed: {result_str}")
            return NodeStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return NodeStatus.FAILURE

class ValidationAction(CompatibleAsyncAction):
    """Custom async action node for result validation."""
    
    async def execute_async(self) -> NodeStatus:
        """Execute validation."""
        try:
            processed_data = self.get_shared_data("processed_data")
            if not processed_data:
                logger.error("No processed data available for validation")
                return NodeStatus.FAILURE
            
            result_str = await _validate_results(str(processed_data))
            
            # Store validation results
            self.set_shared_data("validation_results", result_str)
            
            # Return status based on validation
            if "passed" in result_str.lower():
                logger.info("Validation passed")
                return NodeStatus.SUCCESS
            else:
                logger.warning("Validation failed")
                return NodeStatus.FAILURE
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return NodeStatus.FAILURE

# Condition nodes for decision making
def has_valid_data(context: BTExecutionContext) -> bool:
    """Check if we have valid data to process."""
    collected_data = context.shared_memory.get("collected_data")
    return collected_data is not None and "success" in str(collected_data).lower()

def needs_reprocessing(context: BTExecutionContext) -> bool:
    """Check if data needs reprocessing based on quality."""
    validation_results = context.shared_memory.get("validation_results")
    if not validation_results:
        return False
    return "failed" in str(validation_results).lower()

# Specialized agent for data analysis tasks
class DataAnalysisAgent(BTAgent):
    """Specialized agent for handling data analysis tasks."""
    
    def __init__(self):
        super().__init__(
            name="DataAnalysisAgent",
            instructions="""You are a specialized data analysis agent that can:
            1. Analyze complex data requirements
            2. Collect data from various sources
            3. Process and transform data
            4. Validate results and ensure quality
            
            You work systematically through each step and can retry failed operations.""",
            tools=[analyze_requirements, collect_data, process_data, validate_results],
            model="gpt-4o"
        )
    
    def setup_tree(self) -> BTNode:
        """Set up a complex behavior tree for data analysis tasks."""
        
        # Create condition nodes
        has_data_condition = BTConditionNode("has_valid_data", self, has_valid_data)
        needs_reprocess_condition = BTConditionNode("needs_reprocessing", self, needs_reprocessing)
        
        # Create action nodes
        requirements_node = RequirementAnalysisAction("analyze_requirements", self)
        collection_node = DataCollectionAction("collect_data", self)
        processing_node = DataProcessingAction("process_data", self)
        validation_node = ValidationAction("validate_results", self)
        
        # Create decorators (compatible with both old and new BTEngine)
        reliable_collection = create_retry_decorator(collection_node, max_attempts=3)
        reliable_processing = create_retry_decorator(processing_node, max_attempts=2)
        timed_validation = create_timeout_decorator(validation_node, timeout_seconds=10.0)
        
        # Build the behavior tree structure
        main_sequence = SequenceNode("main_workflow", [
            # Step 1: Analyze requirements
            requirements_node,
            
            # Step 2: Data collection and processing pipeline
            SequenceNode("data_pipeline", [
                reliable_collection,
                
                # Conditional processing based on data availability
                SelectorNode("processing_selector", [
                    SequenceNode("normal_processing", [
                        has_data_condition,
                        reliable_processing,
                        timed_validation
                    ]),
                    # Fallback if no valid data
                    BTAgentAction("log_no_data", self)
                ]),
                
                # Reprocessing loop if validation fails
                SelectorNode("reprocessing_selector", [
                    # If validation passed, we're done
                    SequenceNode("validation_success", [
                        BTConditionNode("validation_passed", self, 
                                      lambda ctx: not needs_reprocessing(ctx))
                    ]),
                    # If validation failed, try reprocessing once more
                    SequenceNode("reprocess_data", [
                        needs_reprocess_condition,
                        DataProcessingAction("reprocess_data", self),
                        validation_node
                    ])
                ])
            ])
        ])
        
        return main_sequence

async def demonstrate_complex_task_execution():
    """Demonstrate complex task execution with behavior trees."""
    
    print("=== Compatible Complex Task Execution Demo ===")
    print(f"BTEngine Version: {'NEW' if NEW_BTENGINE else 'OLD (Compatibility Mode)'}")
    print()
    
    # Create the specialized data analysis agent
    data_agent = DataAnalysisAgent()
    
    # Set up initial task parameters
    initial_data = {
        "task_description": "Analyze customer behavior patterns from sales data",
        "data_source": "sales_database",
        "processing_type": "advanced",
        "query_params": {"date_range": "last_30_days", "customer_segment": "premium"}
    }
    
    print("1. Executing complex data analysis task...")
    print(f"Initial parameters: {initial_data}\n")
    
    # Execute the behavior tree
    if NEW_BTENGINE:
        status = await data_agent.execute_tree(initial_data)
    else:
        # For old BTEngine, use sync execution
        status = data_agent.execute_tree_sync(initial_data)
    
    print(f"Task execution completed with status: {status.name}")
    print(f"Tree status: {data_agent.get_tree_status()}\n")
    
    # Display results
    print("=== Execution Results ===")
    context = data_agent.execution_context
    for key, value in context.shared_memory.items():
        print(f"{key}: {value}")
    
    print("\n=== Tool Results ===")
    for tool_name, result in context.tool_results.items():
        print(f"{tool_name}: {result}")
    
    print("\n" + "="*50)

async def main():
    """Main execution function."""
    try:
        await demonstrate_complex_task_execution()
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 