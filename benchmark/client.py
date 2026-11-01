import asyncio
import time

from benchmark.metrics import TimedResult
from scheduler.scheduler import Scheduler


async def send(
        engine: Scheduler,
        request: list[int],
        planned: float
) -> TimedResult:
    await asyncio.sleep(max(0., planned - time.perf_counter()))
    result = engine.call(request)
    return await asyncio.wrap_future(result), planned


async def benchmark(
        engine: Scheduler,
        requests: list[list[int]],
        arrivals: list[float]
) -> list[TimedResult]:
    tasks = []
    t0 = time.perf_counter()
    for request, arrival in zip(requests, arrivals):
        tasks.append(asyncio.create_task(
            send(engine, request, t0 + arrival)
        ))

    return await asyncio.gather(*tasks)
