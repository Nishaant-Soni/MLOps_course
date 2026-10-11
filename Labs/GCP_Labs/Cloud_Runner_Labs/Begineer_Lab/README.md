# Cloud Run Beginner Lab (modified)

Deploy a containerized Flask app to **Google Cloud Run**, then watch it log, monitor and autoscale.

This version modernizes the original lab: a hardened production Dockerfile, a Cloud Run-aware app, a scripted deploy using **Artifact Registry** (Container Registry is deprecated), explicit scaling settings, and a load-test script that proves autoscaling works. See [What I changed](#what-i-changed).

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask app with `/`, `/health`, `/info`, structured JSON logging |
| `Dockerfile` | Production image (Python 3.12, gunicorn, non-root user) |
| `requirements.txt` | Pinned dependencies (Flask, gunicorn) |
| `.dockerignore` | Keeps scripts/docs/git data out of the image |
| `deploy.sh` | Creates the registry repo, builds, pushes and deploys with scaling flags |
| `loadtest.py` | Concurrent requests against `/info`; counts distinct instances |
| `cleanup.sh` | Deletes the service and registry repo to stop charges |

## 1. What to install

Assume a clean machine. You need:

| Tool | Version | Why | Install |
|---|---|---|---|
| Docker Desktop (or Docker Engine) | 24+ (tested with 28.0.4) | Build and run the image | https://docs.docker.com/get-docker/ |
| Python | 3.9+ (tested with 3.12.6) | Only to run `loadtest.py` (standard library only, no pip packages) | https://www.python.org/downloads/ |
| Google Cloud SDK (`gcloud`) | recent | Deploy to Cloud Run | https://cloud.google.com/sdk/docs/install |
| `curl` | any | Call the endpoints | preinstalled on macOS/Linux |
| `bash` | any | Run `deploy.sh` / `cleanup.sh` | preinstalled on macOS/Linux; on Windows use WSL or Git Bash |
| A Google Cloud account | | A project with **billing enabled** | https://console.cloud.google.com/ |

Use `python3` as written below; on Windows the command is usually `python`. Start Docker Desktop and confirm it is running:

```bash
docker --version
docker info          # must print server details, not "Cannot connect to the Docker daemon"
gcloud --version
```

## 2. Get the code

```bash
git clone https://github.com/Nishaant-Soni/MLOps_course.git
cd MLOps_course/Labs/GCP_Labs/Cloud_Runner_Labs/Begineer_Lab
```

## 3. Test locally first (no Google Cloud needed)

```bash
docker build -t hello-world-local .
docker run -d --name hw-test \
  -e PORT=9090 -e GREETING="Hello from my container" \
  -p 9090:9090 hello-world-local
```

Using `PORT=9090` (not 8080) confirms the container honors the `PORT` variable Cloud Run injects. No `--platform` flag is needed locally: the Dockerfile is OS- and CPU-agnostic and builds natively on Intel/AMD or ARM (Apple Silicon) machines, on macOS, Linux and Windows.

```bash
curl localhost:9090/          # Hello from my container
curl localhost:9090/health    # {"status":"ok"}
curl localhost:9090/info      # {"greeting":"...","instance_id":"fb6590c7","revision":"local","service":"local"}
python3 loadtest.py http://localhost:9090 --requests 40 --concurrency 20 --delay-ms 200
docker logs hw-test           # one JSON log line per request
docker exec hw-test id        # uid=1000(appuser), not root
docker rm -f hw-test          # clean up
```

**A successful local run looks like:**
- `/` returns your greeting, `/health` returns `{"status":"ok"}`.
- `/info` returns JSON with an 8-character `instance_id`.
- The load test prints `0 errors` and `Distinct instances that answered: 1` (locally there is only one container).
- Logs are JSON lines such as `{"severity": "INFO", "message": "request handled", "path": "/info", "status": 200, "latency_ms": 201.72, ...}`.
- `id` shows `appuser`.

## 4. Deploy to Cloud Run

1. Log in (opens a browser):
   ```bash
   gcloud auth login
   ```
2. Create a Google Cloud project in the [console](https://console.cloud.google.com/) and link a billing account to it (required to enable Cloud Run and Artifact Registry). Note its **project ID** (not the display name).
3. Deploy:
   ```bash
   PROJECT_ID=your-project-id ./deploy.sh
   ```
   `deploy.sh` enables the Cloud Run and Artifact Registry APIs, creates the registry repo, builds the image for `linux/amd64`, pushes it, and deploys with these scaling settings (override any via environment variables):

   | Variable | Default | Meaning |
   |---|---|---|
   | `REGION` | `us-east1` | Where to deploy |
   | `SERVICE` | `hello-world` | Cloud Run service name |
   | `MIN_INSTANCES` | `0` | Scale to zero when idle |
   | `MAX_INSTANCES` | `3` | Upper bound on instances |
   | `CONCURRENCY` | `10` | Simultaneous requests per instance |
   | `GREETING` | `Hello from Cloud Run!` | Message returned by `/` |

   Example: `PROJECT_ID=your-project-id MAX_INSTANCES=2 CONCURRENCY=5 ./deploy.sh`

**A successful deploy ends with:**
```
Deployed: https://hello-world-xxxxxxxx-ue.a.run.app
Try:      curl https://hello-world-xxxxxxxx-ue.a.run.app/health
```

## 5. Verify and observe autoscaling

```bash
URL=https://hello-world-xxxxxxxx-ue.a.run.app   # the URL printed by deploy.sh
curl $URL/                # Hello from Cloud Run!
curl $URL/health          # {"status":"ok"}
curl $URL/info            # real service and revision names from Cloud Run
python3 loadtest.py $URL --requests 200 --concurrency 25 --delay-ms 500 --insecure
```

`--insecure` skips TLS certificate verification. It is included because many Python installs (for example python.org builds on macOS) ship without root certificates and would otherwise fail with `CERTIFICATE_VERIFY_FAILED` even though `curl` works. It is acceptable here because you are only calling your own throwaway demo service. If you prefer verification, drop the flag and see the note below.

With `CONCURRENCY=10`, 25 simultaneous slow requests cannot fit on one instance, so Cloud Run starts more. Real output from my deployment:

```
200 requests in 6.9s (28.8 req/s), 0 errors
Distinct instances that answered: 3
  b9402b13: 102 requests
  b5e700d0: 69 requests
  3fa84324: 29 requests
```

More than one distinct instance means Cloud Run scaled out (never more than `MAX_INSTANCES`, which is 3 here). Exact numbers vary.

**To run the load test with certificate verification instead:** remove `--insecure` and first run `pip install certifi` followed by `export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")`, or, for python.org installs on macOS, run `Install Certificates.command` from `/Applications/Python 3.x/`. Without `--insecure` and without trusted certificates, the test prints `200 errors` with `CERTIFICATE_VERIFY_FAILED`.

Then look at it in the console (**Cloud Run > hello-world > Metrics**) for request count, latency and instance count. Under **Logs**, filter on `jsonPayload.path="/info"` to see the structured fields (`latency_ms`, `instance_id`, `status`).

## 6. Clean up

With `MIN_INSTANCES=0`, Cloud Run charges only while handling requests (and the free tier covers light use), but Artifact Registry storage is billed separately. Delete everything when you finish:

```bash
PROJECT_ID=your-project-id ./cleanup.sh
```

## What I changed

The original lab used `python:3.8-slim`, the Flask development server, the deprecated Container Registry, and console clicks for deployment, with scaling and monitoring only mentioned in passing.

1. **Hardened Dockerfile (core change)**
   - Python 3.8 (end-of-life) became `python:3.12-slim`.
   - Dependencies are pinned in `requirements.txt` and installed before the code is copied, so Docker caches that layer.
   - The Flask dev server became **gunicorn** (1 worker, 8 threads) so one instance serves concurrent requests.
   - The container runs as a **non-root user**.
   - The app honors Cloud Run's `$PORT`, and gunicorn is launched in JSON `CMD` form so it receives `SIGTERM` and shuts down gracefully.
   - A `.dockerignore` keeps the image small.
2. **Cloud Run-aware app**
   - `/health` is a health-check endpoint.
   - `/info` reports `K_SERVICE`, `K_REVISION` (set by Cloud Run) and a per-process `instance_id`. An optional `?delay_ms=` (capped at 2000) simulates slow work.
   - Each request is logged as a **structured JSON line**, which Cloud Logging turns into queryable fields.
   - The greeting is configurable through the `GREETING` env var.
3. **Scripted, reproducible deployment**
   - `deploy.sh` replaces the console clicks and the deprecated Container Registry (`gcr.io`) with **Artifact Registry**.
   - It builds with `--platform linux/amd64` only at deploy time (an image built on an Apple Silicon Mac fails on Cloud Run otherwise). The Dockerfile itself stays platform-neutral, and local builds are native.
   - It sets explicit **min/max instances, concurrency, memory and CPU**.
   - It tags each image with a timestamp, and `cleanup.sh` removes the resources.
4. **Autoscaling demo**
   - `loadtest.py` sends concurrent requests and counts the distinct `instance_id`s that answered. This gives visible evidence that Cloud Run scaled out, where the original lab only said "look at the console."

## Testing status

Step 3 (native local build, run, all endpoints, `PORT` override, non-root user, JSON logs and load test) was run and verified locally with Docker 28.0.4. Steps 4 and 5 were also run against a real Cloud Run deployment in `us-east1`: `/`, `/health` and `/info` responded correctly, and the load test scaled to 3 instances with 0 errors. Anyone following this needs their own Google Cloud account and credentials.
