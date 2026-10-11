import json
import os
import sys
import time
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)

# One ID per container process, so repeated calls to /info reveal how many
# Cloud Run instances are serving traffic.
INSTANCE_ID = uuid.uuid4().hex[:8]
GREETING = os.environ.get("GREETING", "Hello, World!")
MAX_DELAY_MS = 2000


def log(severity, message, **fields):
    """Write one JSON line to stdout; Cloud Logging parses it into a structured entry."""
    print(json.dumps({"severity": severity, "message": message, **fields}), file=sys.stdout, flush=True)


@app.before_request
def start_timer():
    request.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    log(
        "INFO",
        "request handled",
        path=request.path,
        method=request.method,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - request.start_time) * 1000, 2),
        instance_id=INSTANCE_ID,
    )
    return response


@app.route("/")
def hello_world():
    return GREETING


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/info")
def info():
    """Report which Cloud Run service/revision/instance answered.

    Optional ?delay_ms=N simulates slow work (capped) so a load test can
    keep requests in flight long enough to trigger autoscaling.
    """
    try:
        delay_ms = min(max(int(request.args.get("delay_ms", 0)), 0), MAX_DELAY_MS)
    except ValueError:
        delay_ms = 0
    time.sleep(delay_ms / 1000)
    return jsonify(
        service=os.environ.get("K_SERVICE", "local"),
        revision=os.environ.get("K_REVISION", "local"),
        instance_id=INSTANCE_ID,
        greeting=GREETING,
    )


if __name__ == "__main__":
    # Local development only; the container runs gunicorn (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
