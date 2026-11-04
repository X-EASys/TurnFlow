"""Local JSON persistence and credential redaction."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Redactor:
    def __init__(self, secrets=()):
        self.secrets = sorted({value for value in secrets if value}, key=len, reverse=True)

    def text(self, value):
        result = str(value)
        for secret in self.secrets:
            result = result.replace(secret, "[REDACTED]")
        result = re.sub(r"\bsk[-_][A-Za-z0-9_-]{12,}", "[REDACTED]", result)
        return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED]", result)

    def json(self, value):
        return json.loads(self.text(json.dumps(value, ensure_ascii=False, default=str)))
