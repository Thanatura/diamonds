import asyncio
import time
import httpx

URL = "http://localhost:8888/hello-async"
PARAMS = {"name": "World"}

async def worker(client, n):
    for _ in range(n):
        r = await client.get(URL, params=PARAMS)
        r.raise_for_status()

async def main(total=1000, concurrency=100):
    per_worker = total // concurrency
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            worker(client, per_worker) for _ in range(concurrency)
        ])
        dt = time.perf_counter() - t0
    print(f"{total} requests in {dt:.2f}s => {total/dt:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(main())