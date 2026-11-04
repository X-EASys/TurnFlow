"""OpenAI-compatible transport with explicit roles and bounded technical retries."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading
import time
from urllib.parse import urlsplit

import httpx

from benchmark.io import Redactor, digest, now, read, save

LOOPBACK = ("127.0.0.1", "localhost", "::1")
DATE_SUFFIX = re.compile(r"-(?:(?:ga|release)-)?(?:\d{4}|\d{6}|\d{8}|\d{4}-\d{2}-\d{2})$")


def family_ok(requested, actual, allowed=()):
    if not isinstance(actual, str):
        return False
    requested = requested.rsplit("/", 1)[-1].lower()
    actual = actual.rsplit("/", 1)[-1].lower()
    if actual == requested:
        return True
    if not actual.startswith(requested):
        return False
    suffix = actual[len(requested):]
    return bool(DATE_SUFFIX.fullmatch(suffix) or
                (actual in {value.rsplit("/", 1)[-1].lower() for value in allowed}
                 and re.fullmatch(r"-(?:ga|release|latest|\d{3})", suffix)))


def routes_from_file(path, agent_model=None):
    document = read(path)
    if set(document) - {"defaults", "agent", "simulator", "judge"}:
        raise ValueError("unknown_config_field")
    defaults = document.get("defaults", {})
    if set(defaults) - {"api_base", "api_key_env", "timeout", "retries"}:
        raise ValueError("unknown_transport_default")
    fixed = {"simulator": "gemini-3.7-flash", "judge": "deepseek-v4-flash"}
    result = {}
    allowed = {"model", "api_base", "api_key_env", "timeout", "retries",
               "allowed_response_models", "reasoning_effort", "omit_temperature"}
    for role in ("agent", "simulator", "judge"):
        item = document.get(role, {})
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("unknown_route_field:" + role)
        route = dict(timeout=1200, retries=3, allowed_response_models=[],
                     reasoning_effort=None, omit_temperature=False)
        route.update(defaults)
        route.update(item)
        if role in fixed:
            if route.get("model", fixed[role]) != fixed[role]:
                raise ValueError("this_release_requires_frozen_" + role + "_model")
            route["model"] = fixed[role]
            route["reasoning_effort"] = "high" if role == "judge" else None
            route["omit_temperature"] = role == "judge"
        if role == "agent" and agent_model:
            route["model"] = agent_model
        if not isinstance(route.get("model"), str) or not re.fullmatch(r"[A-Za-z0-9_./:-]+", route["model"]):
            raise ValueError("valid_model_required:" + role)
        endpoint = route.get("api_base", "")
        if not isinstance(endpoint, str):
            raise ValueError("api_base_must_be_string")
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or any(char.isspace() for char in endpoint)):
            raise ValueError("invalid_endpoint:" + role)
        if parsed.scheme == "http" and parsed.hostname not in LOOPBACK:
            raise ValueError("remote_endpoint_requires_https")
        env = route.get("api_key_env")
        if env is not None and (not isinstance(env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env)):
            raise ValueError("invalid_api_key_env")
        if parsed.hostname not in LOOPBACK and not env:
            raise ValueError("remote_endpoint_requires_api_key_env")
        if type(route["retries"]) is not int or not 0 <= route["retries"] <= 3:
            raise ValueError("retries_must_be_0_to_3")
        if type(route["timeout"]) not in (int, float) or not 1 <= route["timeout"] <= 1800:
            raise ValueError("timeout_must_be_1_to_1800_seconds")
        if type(route["omit_temperature"]) is not bool:
            raise ValueError("omit_temperature_must_be_boolean")
        if route["reasoning_effort"] not in (None, "none", "minimal", "low", "medium", "high", "xhigh"):
            raise ValueError("invalid_reasoning_effort")
        aliases = route["allowed_response_models"]
        if not isinstance(aliases, list) or any(not family_ok(route["model"], alias, aliases) for alias in aliases):
            raise ValueError("response_alias_must_preserve_model_family")
        route["api_base"] = endpoint.rstrip("/")
        result[role] = route
    return result


def classify_response(code, data, route):
    if not isinstance(data, dict):
        return "malformed_response"
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    provider_error = data.get("error")
    if not provider_error and isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            nested = json.loads(content)
            provider_error = nested.get("error") if isinstance(nested, dict) else None
        except ValueError:
            pass
    error_prefix = isinstance(content, str) and re.match(
        r"^(?:error[:：]|\[error\]|upstream (?:error|request failed)|internal server error|"
        r"this token quota is exhausted|insufficient token quota|you exceeded your current quota|"
        r"(?:this|the) model is no longer available|invalid api key|余额不足|当前路由暂无可用渠道)",
        content.strip(), re.I)
    if code != 200 or provider_error or error_prefix:
        text = str(provider_error or content or data).lower()
        if code in (401, 403) or re.search(r"invalid[ _-]?api[ _-]?key|authentication failed|unauthorized", text):
            return "supplier_auth"
        if re.search(r"quota|insufficient[_ ](?:balance|credit)|credit balance|余额不足|额度不足", text):
            return "supplier_quota"
        if re.search(r"no[ _-]longer[ _-]available|model.{0,100}(?:not found|does not exist|unavailable)|no available (?:channel|provider)|暂无可用渠道", text):
            return "channel_model_unavailable"
        return "http_error" if code != 200 else "embedded_provider_error"
    if not isinstance(content, str) or not content.strip():
        return "empty_completion"
    if not family_ok(route["model"], data.get("model"), route["allowed_response_models"]):
        return "response_model_mismatch"
    if first.get("finish_reason") != "stop":
        return "invalid_finish_reason"
    return None


def native_routes(routes, port):
    result = {}
    for role, route in routes.items():
        logical = "turnflow-agent" if role == "agent" else route["model"]
        item = dict(provider="openai", model_name=route["model"],
                    api_base=f"http://127.0.0.1:{port}/{role}/v1",
                    api_key="local-router-no-auth", timeout=route["timeout"] * (route["retries"] + 1) + 120,
                    max_tokens=16384, max_retries=0, extra_body={"stream": False})
        if role == "simulator":
            item["preserve_sampling_parameters"] = True
        if route["omit_temperature"]:
            item["no_temperature"] = True
        if route["reasoning_effort"] is not None:
            item["extra_body"]["reasoning_effort"] = route["reasoning_effort"]
        result[logical] = item
    return {"models": result}


class Gateway:
    """Record every physical attempt without putting credentials in run files."""
    def __init__(self, directory, routes, aims, stop=None):
        self.directory, self.routes = directory, routes
        self.stop = stop or threading.Event()
        self.limits = {"simulator": 1 + 2 * aims, "agent": 1 + aims, "judge": 1}
        self.counts = dict.fromkeys(self.limits, 0)
        self.records = []
        self.lock = threading.Lock()
        self.secrets = {r["api_key_env"]: os.environ.get(r["api_key_env"], "")
                        for r in routes.values() if r.get("api_key_env")}
        missing = [name for name, value in self.secrets.items() if not value]
        if missing:
            raise ValueError("missing_api_key_environment_variables:" + ",".join(missing))
        self.redactor = Redactor(self.secrets.values())

    def call(self, role, body):
        if role not in self.routes:
            raise ValueError("unknown_role")
        route = self.routes[role]
        if (not isinstance(body, dict) or body.get("model") != route["model"]
                or body.get("stream") is not False or not body.get("messages")
                or body.get("max_tokens", body.get("max_completion_tokens")) != 16384):
            raise ValueError("invalid_native_request")
        if set(body) & {"api_key", "authorization", "headers", "prompt_cache_key"}:
            raise ValueError("credential_field_in_body")
        if route["omit_temperature"] and "temperature" in body:
            raise ValueError("temperature_must_be_omitted")
        if route["reasoning_effort"] != body.get("reasoning_effort"):
            raise ValueError("reasoning_effort_mismatch")
        if role == "simulator" and (body.get("temperature") not in (0, 0.7) or "thinking" in body):
            raise ValueError("invalid_simulator_sampling")
        with self.lock:
            if self.stop.is_set() or self.counts[role] >= self.limits[role]:
                return 503, {"error": {"message": "stopped_or_call_limit_reached"}}
            self.counts[role] += 1
            slot = sum(self.counts.values())
        headers = {"Content-Type": "application/json"}
        if route.get("api_key_env"):
            headers["Authorization"] = "Bearer " + self.secrets[route["api_key_env"]]
        error = "transport_error"
        for attempt in range(route["retries"] + 1):
            if self.stop.is_set():
                break
            record = dict(slot=slot, attempt=attempt + 1, role=role, started_at=now(),
                          request_sha256=digest(body), request=body, accepted=False)
            path = self.directory / "calls" / f"{slot:03d}-{role}-{attempt + 1}.json"
            save(path, self.redactor.json(record))
            code = None
            try:
                with httpx.Client(timeout=httpx.Timeout(route["timeout"], connect=min(15, route["timeout"])),
                                  follow_redirects=False,
                                  trust_env=urlsplit(route["api_base"]).hostname not in LOOPBACK) as client:
                    response = client.post(route["api_base"] + "/chat/completions", json=body, headers=headers)
                code = response.status_code
                try:
                    data = json.loads(response.text)
                except ValueError:
                    data = {"error": {"message": "non_json_response"}}
                error = classify_response(code, data, route)
                record.update(http_status=code, response=data, accepted=error is None)
            except Exception as exc:
                error = "transport_error"
                record.update(error_type=type(exc).__name__, error=self.redactor.text(str(exc)))
            record.update(finished_at=now(), error_kind=error)
            save(path, self.redactor.json(record))
            self.records.append(record)
            if error is None:
                return 200, data
            retryable = (error not in ("supplier_auth", "supplier_quota", "channel_model_unavailable", "invalid_finish_reason")
                         and (code is None or code == 200 or code in (408, 409, 429, 500, 502, 503, 504)))
            if not retryable or attempt == route["retries"]:
                break
            if self.stop.wait(min(2 ** attempt, 15)):
                break
        return 502, {"error": {"message": "recorded_technical_failure:" + error}}

    @contextmanager
    def serve(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    pieces = self.path.strip("/").split("/")
                    if len(pieces) != 4 or pieces[1:] != ["v1", "chat", "completions"]:
                        raise ValueError("unknown_gateway_path")
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError("invalid_request_length")
                    code, data = gateway.call(pieces[0], json.loads(self.rfile.read(length)))
                except Exception as exc:
                    code, data = 400, {"error": {"message": gateway.redactor.text(str(exc))}}
                payload = json.dumps(data, ensure_ascii=False).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.block_on_close = False
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)
