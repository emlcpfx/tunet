"""Throwaway: does the Spark agent auto-sync /output to ShareSync (v1.18 §13.6),
or do we still need comfy_run.py's self-upload workaround? Submits a tiny
busybox job that writes one file to /output (NO self-upload), waits for terminal,
then PROPFINDs the output folder. Also validates the new /api/auth/login path.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy_launch as cl

print("[test] logging in via", cl.SPARK_API + "/api/auth/login ...")
tok = cl.get_token()
print("[test] auth OK, token prefix:", tok[:18], "...")

body = {
    "instanceType": "g4dn.xlarge",
    "image": "busybox:latest",
    "command": ["sh", "-c",
                "echo hello-from-spark > /output/autosync_test.txt && "
                "echo wrote: && ls -la /output/"],
    "tags": [cl.BILLING_TAG, cl.GROUP_TAG, "autosync-test"],
    "idleHoldSeconds": 0,
    "mode": "instant",
}
print("[test] submitting no-input busybox job (g4dn.xlarge, idleHold=0)...")
resp = cl.spark("POST", "/api/compute/jobs", body).json()
job_id = resp.get("jobId") or resp.get("id")
out_url = (resp.get("output") or {}).get("shareSyncBaseUrl")
print("[test] jobId:", job_id)
print("[test] output.shareSyncBaseUrl:", out_url)

print("[test] streaming logs to terminal...")
cl.stream_logs(job_id)

print("[test] confirming terminal status...")
status = cl.wait_for_terminal(job_id)
detail = cl.get_job(job_id)
print(f"[test] terminal status: {status}  exit_code={detail.get('exit_code')}  "
      f"error_code={detail.get('error_code')}")

# Resolve the output URL (submit resp first; else rebuild from job detail).
if not out_url:
    out_url = (detail.get("output") or {}).get("shareSyncBaseUrl")
print("[test] PROPFIND output folder:", out_url)

found = False
if out_url:
    for attempt in range(6):  # retry ~30s in case the drain lags the terminal flip
        try:
            entries = cl.webdav_list(out_url)
        except Exception as e:  # noqa: BLE001
            print(f"[test]   PROPFIND error: {e}")
            entries = []
        names = [n for n, _, _ in entries]
        print(f"[test]   attempt {attempt+1}: {names or '(empty)'}")
        if any("autosync_test.txt" in n for n in names):
            found = True
            break
        time.sleep(5)
else:
    print("[test] no output URL to check.")

print()
print("=" * 60)
if found:
    print("RESULT: agent DID auto-sync /output -> ShareSync.")
    print("  => comfy_run.py self-upload is now REDUNDANT (per v1.18 §13.6).")
else:
    print("RESULT: autosync_test.txt did NOT appear in ShareSync.")
    print("  => agent still does NOT sync /output; self-upload workaround STAYS.")
print("=" * 60)
