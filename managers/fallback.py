import asyncio
from models.fallback import FallbackRequest, FallbackStrategy
from models.common import BaseRequest
from typing import Callable, Awaitable, TypeAlias, Any, List, Set, Tuple
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
        return await asyncio.wait_for(executor(request), timeout)
    
    async def execute_sequential(
        self, 
        requests: List[BaseRequest], 
        executor: Executor, 
        timeout: int = 360
    ) -> Any:
        """
        tries to execute the request with the executors sequentially.
        if an executor fails, it will try the next executor.
        if all executors fail, it will raise an exception.
        the first executor to succeed will be returned.
        """
        for request in requests:
            try:
                result = await self.execute(request, executor, timeout)
                logfire.info(f"Succeded request: {request}")
                return result
            except Exception as e:
                logfire.warning(f"Request {request} failed: {e}")
                continue
        raise Exception("All requests failed")
    
    async def execute_parallel(self, requests: List[BaseRequest], executor: Executor, timeout: int = 360) -> Any:
        """
        tries all executors in parallel
        the first one to succeed will be returned.
        if all executors fail, it will raise an exception.
        """
        tasks = [self.execute(request, executor, timeout) for request in requests]
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout
        )
        done: Set[asyncio.Task[Any]] = done
        if not done:
            logfire.error("All executors failed")
            raise Exception("All executors failed")
        for pending_task in pending:
            pending_task: asyncio.Task[Any] = pending_task
            pending_task.cancel()
        done_task: asyncio.Task[Any] = done.pop()
        return done_task.result()
    

    async def catch(self, request: FallbackRequest, executor: Executor) -> Any:
        if request.strategy == FallbackStrategy.SEQUENTIAL:
            return await self.execute_sequential(request.requests, executor, request.timeout_per_request)
        elif request.strategy == FallbackStrategy.PARALLEL:
            return await self.execute_parallel(request.requests, executor, request.timeout_per_request)
        else:
            raise ValueError(f"Invalid strategy: {request.strategy}")