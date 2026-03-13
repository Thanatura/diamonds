import time
import requests

URL = "http://localhost:8888/hello"

def main(n=100):
    t0 = time.perf_counter()
    for _ in range(n):
        r = requests.get(URL, params={"name": "World"})
        r.raise_for_status()
    dt = time.perf_counter() - t0
    print(f"{n} requests in {dt:.2f}s => {n/dt:.1f} req/s")

if __name__ == "__main__":
    main(1000)