"""The bundled, versioned 490-scenario dataset. Loading is entirely offline."""
from collections import Counter
import hashlib
from importlib.resources import files
import os
from pathlib import Path

from benchmark.io import read

TASKS = ("T1", "T2", "T3", "T4", "T5")
ROOT = Path(str(files("benchmark").joinpath("resources/dataset")))


def protocol():
    return read(ROOT / "protocol.json")


def bindings():
    return read(ROOT / "bindings.json")["registrations"]


def source_path(key):
    if key not in protocol()["scenario_keys"]:
        raise ValueError("unknown_source_key:" + str(key))
    return ROOT / "sources" / (key + ".json")


def load(key):
    from benchmark.data_generation.scenario_loader import ScenarioLoader
    return ScenarioLoader().load_from_file(source_path(key))


def select(*, tasks=None, keys=None, smoke=None):
    candidates = protocol()["scenario_keys"]
    if tasks:
        if set(tasks) - set(TASKS):
            raise ValueError("unknown_task")
        candidates = [key for key in candidates if key.split("/")[0] in tasks]
    if keys:
        if len(set(keys)) != len(keys) or not set(keys) <= set(candidates):
            raise ValueError("source_keys_outside_selection_or_duplicate")
        candidates = [key for key in candidates if key in keys]
    if smoke is not None:
        if smoke < 1 or keys:
            raise ValueError("smoke_requires_positive_count_and_no_explicit_keys")
        candidates = [key for task in TASKS for key in
                      [key for key in candidates if key.startswith(task + "/")][:smoke]]
    if not candidates:
        raise ValueError("empty_selection")
    return candidates


def verify():
    """Verify source bytes, model loading, simulator bindings and judge compilation."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from benchmark.evaluator.task_entry import compile_spec, get_catalog, verify_source
    from benchmark.user.release_profile import create_release_simulator
    from loguru import logger
    logger.disable("benchmark")

    manifest = read(ROOT / "manifest.json")
    package = Path(str(files("benchmark")))
    for relative, expected in manifest["files"].items():
        actual = hashlib.sha256((package / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("release_integrity_mismatch:" + relative)
    conf = protocol()
    keys = conf["scenario_keys"]
    if len(keys) != 490 or len(set(keys)) != 490 or set(keys) != set(get_catalog()):
        raise ValueError("dataset_catalog_mismatch")
    registration = bindings()
    for key in keys:
        path = source_path(key)
        verify_source(key, read(path))
        scenario = load(key)
        simulator = create_release_simulator(
            scenario, conf["simulator_model"], source_path=path,
            generation_candidate=conf["generation_candidate"],
            candidate_source_key=key, candidate_bindings=registration,
            candidate_strict=conf["candidate_strict"])
        if simulator.simulator_identity["source_key"] != key:
            raise ValueError("simulator_source_mismatch:" + key)
        spec = compile_spec(key.split("/")[0], key, [
            {"role": "user", "content": "Please help with the request."},
            {"role": "assistant", "content": "I will check the constraints first."},
        ], conf["judge_model"], runtime_profile=conf["judge_runtime_profile"])
        if spec["request"]["max_tokens"] != conf["max_tokens"]:
            raise ValueError("judge_token_budget_mismatch:" + key)
    counts = dict(Counter(key.split("/")[0] for key in keys))
    if counts != conf["counts"]:
        raise ValueError("dataset_counts_mismatch")
    return {"version": conf["version"], "sources": len(keys), "counts": counts,
            "integrity_files": len(manifest["files"]), "model_calls": 0}
