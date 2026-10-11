"""Fire concurrent requests at /info and count the distinct instances that answered.

Usage: python loadtest.py URL [--requests 200] [--concurrency 50] [--delay-ms 500] [--insecure]
Uses only the standard library.
"""
import argparse
import json
import ssl
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def call(url, context):
    try:
        with urllib.request.urlopen(url, timeout=30, context=context) as resp:
            return json.load(resp)["instance_id"]
    except Exception as exc:
        return f"ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Base URL, e.g. https://hello-world-xxxx.a.run.app")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--delay-ms", type=int, default=500)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS certificate verification (workaround for Python installs missing root certificates)",
    )
    args = parser.parse_args()

    context = ssl._create_unverified_context() if args.insecure else None

    target = f"{args.url.rstrip('/')}/info?delay_ms={args.delay_ms}"
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda u: call(u, context), [target] * args.requests))
    elapsed = time.perf_counter() - start

    counts = Counter(results)
    errors = sum(n for key, n in counts.items() if key.startswith("ERROR"))
    instances = {key: n for key, n in counts.items() if not key.startswith("ERROR")}

    print(f"{args.requests} requests in {elapsed:.1f}s ({args.requests / elapsed:.1f} req/s), {errors} errors")
    print(f"Distinct instances that answered: {len(instances)}")
    for instance, n in sorted(instances.items(), key=lambda kv: -kv[1]):
        print(f"  {instance}: {n} requests")
    if errors:
        sample = next(key for key in counts if key.startswith("ERROR"))
        print(f"Sample error: {sample}")
        if "CERTIFICATE_VERIFY_FAILED" in sample:
            print("Hint: your Python lacks root certificates. Re-run with --insecure, or on macOS run "
                  "'Install Certificates.command' from /Applications/Python 3.x/.")


if __name__ == "__main__":
    main()
