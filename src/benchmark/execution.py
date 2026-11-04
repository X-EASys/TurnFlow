"""Batch execution and reporting. Run directories are local, append-only outputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from benchmark import dataset
from benchmark.io import digest, now, read, save
from benchmark.transport import Gateway, native_routes, routes_from_file

DECISIONS = ("PASS", "FAIL", "U", "NA", "INVALID", "notrun")


def position_path(output, key):
    return output / "positions" / (key.replace("/", "__") + ".json")


def code_fingerprint():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob("*.py"))})


def report(output):
    output = Path(output).resolve()
    manifest = read(output / "run.json")
    rows = [read(position_path(output, key)) for key in manifest["selection"]]
    counts = Counter(row["decision_status"] for row in rows)
    pf = counts["PASS"] + counts["FAIL"]
    planned = len(rows)
    result = {
        "version": manifest["version"], "agent_model": manifest["routes"]["agent"]["model"],
        "planned": planned, "counts": {key: counts[key] for key in DECISIONS},
        "complete_dialogues": sum(row.get("conversation_complete", False) for row in rows),
        "pass_rate": counts["PASS"] / pf if pf else None,
        "scoreable_coverage": pf / planned, "planned_pass_rate": counts["PASS"] / planned,
        "physical_attempts": sum(row.get("physical_attempts", 0) for row in rows),
        "by_task": {task: dict(Counter(row["decision_status"] for row in rows
                                      if row["source_key"].startswith(task + "/")))
                    for task in dataset.TASKS},
    }
    save(output / "summary.json", result)
    with (output / "positions.csv").open("w", newline="", encoding="utf-8") as handle:
        os.chmod(output / "positions.csv", 0o600)
        writer = csv.DictWriter(handle, fieldnames=["source_key", "status", "decision_status",
                                                   "conversation_complete", "physical_attempts"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    return result


def run(config, output, *, agent_model=None, tasks=None, keys=None, smoke=None,
        concurrency=1, resume=False, prepare_only=False):
    if not 1 <= concurrency <= 32:
        raise ValueError("concurrency_must_be_1_to_32")
    output = Path(output).expanduser().resolve()
    package = Path(__file__).resolve().parent
    if output == package or package in output.parents:
        raise ValueError("run_output_must_be_outside_installed_package")
    routes = routes_from_file(config, agent_model)
    selection = dataset.select(tasks=tasks, keys=keys, smoke=smoke)
    dataset.verify()
    identity = dict(version="0.4.0", routes=routes, selection=selection,
                    dataset_manifest_sha256=hashlib.sha256((dataset.ROOT / "manifest.json").read_bytes()).hexdigest(),
                    code_sha256=code_fingerprint())
    if (output / "run.json").exists():
        if not resume:
            raise ValueError("existing_run_requires_resume_or_new_output_directory")
        if read(output / "run.json")["identity"] != digest(identity):
            raise ValueError("resume_code_data_config_or_selection_mismatch")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("new_run_requires_empty_output_directory")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        save(output / "run.json", dict(identity= digest(identity), created_at=now(), **identity))
        for key in selection:
            save(position_path(output, key), {"source_key": key, "status": "planned", "decision_status": "notrun"})
    if prepare_only:
        return report(output)
    missing = sorted({route["api_key_env"] for route in routes.values()
                      if route.get("api_key_env") and not os.environ.get(route["api_key_env"])})
    if missing:
        raise ValueError("missing_api_key_environment_variables:" + ",".join(missing))
    # Only one process may own a run directory; Linux/macOS use an OS lock.
    import fcntl
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("run_directory_is_already_active") from exc
        stop = threading.Event()
        active, active_lock = {}, threading.Lock()
        for key in selection:
            row = read(position_path(output, key))
            if row["status"] == "running":
                row.update(status="interrupted", decision_status="INVALID", finished_at=now(),
                           error="previous_process_ended_after_reservation")
                save(position_path(output, key), row)

        def worker(key):
            path = position_path(output, key)
            row = read(path)
            if row["status"] != "planned" or stop.is_set():
                return
            directory = output / "runs" / key.replace("/", "__")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            row.update(status="running", decision_status="INVALID", started_at=now())
            save(path, row)
            aims = len(dataset.load(key).state_queue.aim_sequence.aims)
            gateway = Gateway(directory, routes, aims, stop)
            try:
                with gateway.serve() as port:
                    save(directory / "native-routes.json", native_routes(routes, port))
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
                    # Credentials live only in the parent gateway's memory.
                    for name in gateway.secrets:
                        env.pop(name, None)
                    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
                                 "PYTHONPATH", "PYTHONHOME", "BENCHMARK_LLM_ROUTES"):
                        env.pop(name, None)
                    bypass = {value.strip() for name in ("NO_PROXY", "no_proxy")
                              for value in env.get(name, "").split(",") if value.strip()}
                    bypass.update(("127.0.0.1", "localhost", "::1"))
                    env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(bypass))
                    command = [sys.executable, "-I", "-B", "-m", "benchmark.worker",
                               "--directory", str(directory), "--source-key", key]
                    process = subprocess.Popen(command, env=env, cwd=directory,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, errors="replace", start_new_session=True)
                    with active_lock:
                        active[key] = process
                    with (directory / "runtime.log").open("w", encoding="utf-8") as log:
                        os.chmod(directory / "runtime.log", 0o600)
                        for line in process.stdout:
                            log.write(gateway.redactor.text(line))
                            log.flush()
                    process.stdout.close()
                    returncode = process.wait()
                    with active_lock:
                        active.pop(key, None)
                native_path = directory / "native-result.json"
                native = read(native_path) if native_path.exists() else {}
                complete = native.get("simulation", {}).get("termination_reason") == "success"
                decision = native.get("decision_status", "INVALID")
                accepted = returncode == 0 and complete and decision in ("PASS", "FAIL", "U", "NA")
                row.update(status="completed" if accepted else "failed", conversation_complete=complete,
                           decision_status=decision if accepted else "INVALID", native_returncode=returncode,
                           native_result=str(native_path.relative_to(output)) if native_path.exists() else None)
                if native_path.exists():
                    save(native_path, gateway.redactor.json(native))
            except Exception as exc:
                row.update(status="failed", decision_status="INVALID", error_type=type(exc).__name__,
                           error=gateway.redactor.text(str(exc)))
            row.update(finished_at=now(), physical_attempts=len(gateway.records), logical_calls=dict(gateway.counts))
            save(path, row)
            final_by_slot = {}
            for record in gateway.records:
                final_by_slot[record["slot"]] = record
            if any(record.get("error_kind") in ("supplier_auth", "supplier_quota", "channel_model_unavailable", "response_model_mismatch")
                   for record in final_by_slot.values()):
                stop.set()
            print(f"{key}: {row['decision_status']} ({row['physical_attempts']} API attempts)", flush=True)

        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            futures = [executor.submit(worker, key) for key in selection]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            with active_lock:
                processes = list(active.values())
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            result = report(output)
        return result
