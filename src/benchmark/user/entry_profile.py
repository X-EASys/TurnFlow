"""Exact-source integration of the frozen O1 / R2 / R2 / O3 renderers.

This registry selects inputs, never classifies generated content. Each source
has explicit optional opening and generation-stage bindings. The original
scenario remains available to the controller; generation bindings live only in
this renderer's private copy. No model call, repair or retry is introduced.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.resources import files
import json

from benchmark.user.opening_prompt_options import build_opening_messages
from benchmark.user.preference_midturn_options import build_midturn_messages
from benchmark.user.preference_prompt_options import build_preference_option
from benchmark.user.typed_generation_boundary import (
    bind_preference_source, validated_current_source,
)


revision = 'simulator-o1-r2-o3-priority-repair-20261029-v4'
_RESOURCE_SHA256 = '7b29bf84faa65643285f5b02d0d0bedc5dd393380e116a0a49035b165c8a473b'
_P1_HEADER = "PREFERENCE ALREADY EXPRESSED — GENERATION BOUNDARY, NOT NEW SPEECH:"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def _catalog():
    raw = files("benchmark.user").joinpath("resources/simulator_entry_profile.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != _RESOURCE_SHA256:
        raise ValueError("simulator_entry_profile_integrity_mismatch")
    data = json.loads(raw)
    if data["revision"] != revision:
        raise ValueError("simulator_entry_profile_revision_mismatch")
    return data


def manifest():
    """Return the frozen input bindings and prompt provenance."""
    return deepcopy(_catalog())


def select_registration(scenario, source_hash):
    rows = [row for row in _catalog()["bindings"].values()
            if source_hash is not None and row["source_sha256"] == source_hash]
    if len(rows) > 1:
        raise ValueError("ambiguous_simulator_entry_source")
    if not rows:
        return None, "source_not_registered"
    if digest(scenario.model_dump(mode="json")) != rows[0]["scenario_sha256"]:
        return None, "scenario_object_mismatch"
    return deepcopy(rows[0]), "registered_source"


def render_final(private, aim, history, baseline):
    """Validate actual emitted source, accounting for a real P1 note if present.

    A private binding alone must not be mistaken for a note already present in
    the generation request. Exact registered P1 sources can share this path.
    """
    source = validated_current_source(private, aim, baseline)
    released, computed_note = bind_preference_source(private, aim, history)
    if not aim.is_final or not released:
        raise ValueError("entry_final_requires_released_preference")
    old_note = computed_note if _P1_HEADER in baseline[1]["content"] else ""
    return build_preference_option(baseline, option="O3", preference_source=released,
                                  old_preference_note=old_note, **source)


class EntryRenderer:
    """One simulator instance's immutable source and diagnostic request events."""

    def __init__(self, scenario, registration):
        self._source = deepcopy(scenario)
        self._registration = deepcopy(registration)
        self.events = []

    def _record(self, stage, role, variant, baseline, actual, **details):
        row = dict(stage=stage, role=role, prompt_variant=variant,
                   source_sha256=self._registration["source_sha256"],
                   scenario_sha256=self._registration["scenario_sha256"],
                   baseline_messages=deepcopy(baseline), actual_messages=deepcopy(actual),
                   baseline_messages_sha256=digest(baseline),
                   actual_messages_sha256=digest(actual), **details)
        self.events.append(row)
        return actual

    def opening(self, baseline, *, anchor_guidance, is_disguised):
        bound = self._registration.get("opening")
        if bound is None:
            return baseline
        if anchor_guidance != bound["opening_intent"]:
            raise ValueError("entry_opening_source_mismatch")
        actual = build_opening_messages(opening_intent=anchor_guidance, speaker=bound["speaker"])
        return self._record("OPENING", "opening", "O1", baseline, actual,
                            aim_id=None, opening_intent=anchor_guidance,
                            anchor_disguised=is_disguised)

    def generation(self, baseline, *, aim, history):
        bound = self._registration.get("generation")
        if bound is None:
            return baseline
        if aim.aim_id == bound.get("preference_aim_id"):
            role, variant = "anchor", "R2"
        elif aim.aim_id in bound["material_aim_ids"]:
            role, variant = "material", "R2"
        elif aim.aim_id == bound.get("final_aim_id") and aim.is_final:
            role, variant = "final", "O3"
        else:
            # Explicit source-authorized updates and unregistered turns retain
            # their normal generation path; roles are never inferred online.
            return baseline
        private = deepcopy(self._source)
        private.metadata = {**(private.metadata or {}),
                            "simulator_no_repeat": deepcopy(bound["preference_binding"])}
        source = validated_current_source(private, aim, baseline)
        actual = (render_final(private, aim, history, baseline) if role == "final" else
                  build_midturn_messages(private, aim, history, baseline, role=role, arm="R2"))
        return self._record("GENERATE", role, variant, baseline, actual,
                            aim_id=aim.aim_id, current_intent=source["current_intent"],
                            current_public_payload=deepcopy(source["current_payload"]),
                            actual_history=[m.model_dump(mode="json") for m in history],
                            baseline_has_p1_note=_P1_HEADER in baseline[1]["content"])
