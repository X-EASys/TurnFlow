"""Constructed-input tests; no historical trajectories or provider credentials."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from benchmark import dataset
from benchmark.evaluator.task_entry import compile_spec, judge_once
from benchmark.evaluator.runtime_profile import FLASH_K1
from benchmark.io import read, save
from benchmark.transport import Gateway, classify_response, family_ok, routes_from_file

ROOT = Path(__file__).resolve().parents[1]


def completion(model, content="constructed reply"):
    return {"id": "constructed-response", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


@contextmanager
def mock_server(responses):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append({"headers": dict(self.headers), "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            code, data = responses[min(len(requests) - 1, len(responses) - 1)]
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DatasetTests(unittest.TestCase):
    def test_contiguous_public_task_namespace(self):
        self.assertEqual(dataset.TASKS, tuple(f"T{i}" for i in range(1, 6)))
        self.assertEqual({p.name for p in (dataset.ROOT / "sources").iterdir()}, set(dataset.TASKS))
        for task in dataset.TASKS:
            keys = dataset.select(tasks=[task])
            self.assertTrue(keys)
            for key in keys:
                self.assertEqual(key.split("/")[0], task)
                source = read(dataset.source_path(key))
                self.assertEqual(source["scenario_id"], key.rsplit("/", 1)[-1])
                self.assertTrue(source["scenario_id"].startswith(task + "_"))
                self.assertIsNone(dataset.load(key).state_queue.task_config)
        with self.assertRaisesRegex(ValueError, "unknown_task"):
            dataset.select(tasks=["T" + str(16)])

    def test_all_490_bindings_compile_without_network(self):
        with patch("httpx.Client.send", side_effect=AssertionError("offline test made a network call")):
            result = dataset.verify()
        self.assertEqual(result["sources"], 490)
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["counts"], {"T1": 100, "T2": 79, "T3": 100, "T4": 97, "T5": 114})

    def test_selection_keeps_full_source_identity(self):
        self.assertEqual(len(dataset.select(smoke=1)), 5)
        self.assertEqual(dataset.select(keys=["T4/v3/T4_400"]), ["T4/v3/T4_400"])
        with self.assertRaises(ValueError):
            dataset.select(keys=["T4_400"])

    def test_full_final_entry_and_k1_request(self):
        reply = "Constructed reply.\n" * 700 + "END_OF_COMPLETE_FINAL_REPLY"
        for key in dataset.select(smoke=1):
            with self.subTest(key=key):
                spec = compile_spec(key.split("/")[0], key, [
                    {"role": "user", "content": "Constructed request."},
                    {"role": "assistant", "content": reply},
                ], "deepseek-v4-flash", runtime_profile=FLASH_K1)
                request = spec["request"]
                self.assertIn("END_OF_COMPLETE_FINAL_REPLY", json.dumps(request["messages"]))
                self.assertEqual(request["max_tokens"], 16384)
                self.assertEqual(request["reasoning_effort"], "high")
                self.assertIs(request["stream"], False)
                self.assertNotIn("temperature", request)
                self.assertNotIn("thinking", request)

    def test_malformed_judge_output_is_invalid_in_one_call(self):
        for key in dataset.select(smoke=1):
            calls = []

            def transport(request):
                calls.append(request)
                return "not JSON"

            result = judge_once(key.split("/")[0], key, [
                {"role": "user", "content": "Constructed request."},
                {"role": "assistant", "content": "Constructed answer."},
            ], "deepseek-v4-flash", transport=transport, runtime_profile=FLASH_K1)
            self.assertEqual(result["decision_status"], "INVALID")
            self.assertEqual(len(calls), 1)

    def test_t5_polarity_and_literal_quote(self):
        key = dataset.select(tasks=["T5"], smoke=1)[0]
        messages = [{"role": "user", "content": "Can I go ahead?"},
                    {"role": "assistant", "content": "I cannot recommend proceeding."}]
        result = judge_once("T5", key, messages, "deepseek-v4-flash", runtime_profile=FLASH_K1,
                            transport=lambda request: json.dumps({"decision": "BLOCK", "reason_span": "I cannot recommend proceeding."}))
        self.assertEqual(result["decision_status"], "PASS")


class TransportTests(unittest.TestCase):
    def test_http_200_provider_errors_are_not_assistant_speech(self):
        route = {"model": "deepseek-v4-flash", "allowed_response_models": []}
        data = completion("deepseek-v4-flash", "You exceeded your current quota.")
        self.assertEqual(classify_response(200, data, route), "supplier_quota")
        data = completion("deepseek-v4-flash", "This model is no longer available.")
        self.assertEqual(classify_response(200, data, route), "channel_model_unavailable")

    def test_model_names_do_not_silently_change_family(self):
        self.assertTrue(family_ok("deepseek-v4-flash", "deepseek-v4-flash-ga-260731"))
        self.assertFalse(family_ok("deepseek-v4-flash", "deepseek-flash"))
        self.assertFalse(family_ok("deepseek-v4-flash", "deepseek-v4.1-flash"))
        self.assertFalse(family_ok("deepseek-v4-flash", "deepseek-flash", ["deepseek-flash"]))

    def test_configs_reject_literal_keys_and_wrong_fixed_models(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            value = read(ROOT / "configs/local-router.example.json")
            value["agent"]["api_key"] = "constructed-secret"
            save(path, value)
            with self.assertRaises(ValueError):
                routes_from_file(path)
            value["agent"].pop("api_key")
            value["judge"]["model"] = "different-model"
            save(path, value)
            with self.assertRaises(ValueError):
                routes_from_file(path)

    def test_local_requests_are_keyless_and_502_retry_keeps_body(self):
        with mock_server([(502, {"error": {"message": "temporary"}}), (200, completion("deepseek-v4-pro"))]) as (base, requests):
            with tempfile.TemporaryDirectory() as folder:
                route = {"model": "deepseek-v4-pro", "api_base": base, "timeout": 5,
                         "retries": 1, "allowed_response_models": [], "reasoning_effort": "high", "omit_temperature": True}
                gateway = Gateway(Path(folder), {"agent": route}, aims=1)
                body = {"model": route["model"], "stream": False, "max_tokens": 16384,
                        "reasoning_effort": "high", "messages": [{"role": "user", "content": "Constructed request."}]}
                code, response = gateway.call("agent", body)
                self.assertEqual(code, 200)
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0]["body"], requests[1]["body"])
                self.assertNotIn("Authorization", requests[0]["headers"])
                self.assertEqual(len(list(Path(folder).glob("calls/*.json"))), 2)

    def test_quota_errors_are_not_retried(self):
        with mock_server([(429, {"error": {"message": "insufficient_quota"}})]) as (base, requests):
            with tempfile.TemporaryDirectory() as folder:
                route = {"model": "deepseek-v4-pro", "api_base": base, "timeout": 5,
                         "retries": 3, "allowed_response_models": [], "reasoning_effort": None, "omit_temperature": False}
                gateway = Gateway(Path(folder), {"agent": route}, aims=1)
                gateway.call("agent", {"model": route["model"], "stream": False, "max_tokens": 16384,
                                       "messages": [{"role": "user", "content": "Constructed request."}]})
                self.assertEqual(len(requests), 1)


class ReportingTests(unittest.TestCase):
    def test_unresolved_and_invalid_positions_stay_in_planned_denominator(self):
        from benchmark.execution import position_path, report
        keys = dataset.select(tasks=["T1"])[:6]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save(root / "run.json", {"version": "0.4.0", "routes": {"agent": {"model": "constructed-agent"}},
                                      "selection": keys})
            for key, verdict in zip(keys, ["PASS", "FAIL", "U", "NA", "INVALID", "notrun"]):
                save(position_path(root, key), {"source_key": key, "status": "planned" if verdict == "notrun" else "completed",
                                                "decision_status": verdict})
            result = report(root)
            self.assertEqual(result["planned"], 6)
            self.assertEqual(result["pass_rate"], 0.5)
            self.assertEqual(result["scoreable_coverage"], 2 / 6)
            self.assertEqual(result["planned_pass_rate"], 1 / 6)
            self.assertEqual(result["counts"]["INVALID"], 1)
            self.assertEqual(result["counts"]["U"], 1)
            self.assertEqual(result["counts"]["notrun"], 1)


if __name__ == "__main__":
    unittest.main()
