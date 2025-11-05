import asyncio
from models.fallback import FallbackRequest, FallbackStrategy
from models.common import BaseRequest
from typing import Callable, Awaitable, TypeAlias, Any, List, Set
import logfire

Executor: TypeAlias = Callable[[BaseRequest], Awaitable[Any]]

class FallbackManager:

    def __init__(self):
        pass


    async def execute(
        self, 
        request: BaseRequest, 
        executor: Executor, 
        timeout: int = 360
    ) -> Any:
        """
        Execute a single request with a timeout.
        
        Args:
            request: The request to execute
            executor: The async function to execute the request
            timeout: Timeout in seconds (not milliseconds)
        
        Returns:
            The result from the executor
        """
        return await asyncio.wait_for(executor(request), timeout)
    
    async def execute_sequential(
        self, 
        requests: List[BaseRequest], 
        executor: Executor, 
        timeout: int = 360
    ) -> Any:
        """
        Tries to execute the request with the executors sequentially.
        If an executor fails, it will try the next executor.
        If all executors fail, it will raise an exception.
        The first executor to succeed will be returned.
        
        Args:
            requests: List of requests to try
            executor: The async function to execute each request
            timeout: Timeout per request in seconds (not milliseconds)
        
        Returns:
            The result from the first successful request
        """
        for request in requests:
            try:
                result = await self.execute(request, executor, timeout)
                logfire.info(f"Succeded request: {request}")
                return result
            except Exception as e:
                logfire.warning(f"Request {request} failed: {e}")
                continue
        logfire.error("All requests failed")
    
    async def execute_parallel(self, requests: List[BaseRequest], executor: Executor, timeout: int = 360) -> Any:
        """
        Tries all executors in parallel.
        The first one to succeed will be returned.
        If all executors fail, it will raise an exception.
        
        Args:
            requests: List of requests to try in parallel
            executor: The async function to execute each request
            timeout: Timeout per request in seconds (not milliseconds)
        
        Returns:
            The result from the first successful request
        """
        # Create tasks (not coroutines) for asyncio.wait()
        # tasks = [asyncio.create_task(self.execute(request, executor, timeout)) for request in requests]
        # pending = set(tasks)

        task_to_request = {asyncio.create_task(self.execute(request, executor, timeout)): request \
                                                                            for request in requests}
        pending = set[asyncio.Task[Any]](task_to_request.keys())
        done: Set[asyncio.Task[Any]] = set()

        
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            # done: Set[asyncio.Task[Any]] = done
            
            # Check each completed task for success
            for done_task in done:
                try:
                    result = done_task.result()
                    # Success! Cancel remaining tasks and return
                    for pending_task in pending:
                        pending_task.cancel()
                    logfire.info(f"First succeeded request in parallel: {task_to_request[done_task]}")
                    return result
                except Exception:
                    # This task failed, log and continue to check other tasks
                    logfire.warning(f"Parallel request failed: {done_task.exception()}")
                    continue
        
        # All tasks completed but all failed
        logfire.error("All parallel executors failed")
    

    async def catch(self, request: FallbackRequest, executor: Executor) -> Any:
        if request.strategy == FallbackStrategy.SEQUENTIAL:
            return await self.execute_sequential(request.requests, executor, request.timeout_per_request)
        elif request.strategy == FallbackStrategy.PARALLEL:
            return await self.execute_parallel(request.requests, executor, request.timeout_per_request)
        else:
            raise ValueError(f"Invalid strategy: {request.strategy}")