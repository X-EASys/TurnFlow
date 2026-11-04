"""Frozen simulator mechanisms, selected independently of model routing.

The source catalog contains only previously studied P1 bindings. Exact file
hashes resolve full scenario identities; a bare scenario ID never enables P1.
Only the simulator's private copy receives its generation configuration.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re

from benchmark.user.aim_driven_simulator import AimDrivenUserSimulator
from benchmark.user.first_pass import CallBudget, FirstPassTrace
from benchmark.user.user_simulator import UserSimulator


revision = "simulator-b1-v3-p1-20261009-v1"
_RESOURCE_SHA256 = '92f8b77cc314c2c3101524b8202d3bcdbf2666e4f1acaa2c0bf3bbffdaa954de'
_TASKS = frozenset(("T1", "T2", "T3", "T4", "T5"))


def _digest_scenario(scenario):
    return hashlib.sha256(json.dumps(scenario.model_dump(mode="json"),
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def _catalog():
    raw = files("benchmark.user").joinpath("resources/release_profile.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != _RESOURCE_SHA256:
        raise ValueError("simulator_profile_integrity_mismatch")
    data = json.loads(raw)
    if data["revision"] != revision:
        raise ValueError("simulator_profile_revision_mismatch")
    return data


def manifest():
    """Return the frozen mechanism and source-binding provenance."""
    return deepcopy(_catalog())


def _task(scenario):
    value = getattr(scenario, "task_type", None)
    if value:
        value = getattr(value, "value", value)
        if str(value).upper() in _TASKS:
            return str(value).upper()
    match = re.match(r"^(T\d+)(?:_|$)", scenario.scenario_id or "", re.I)
    return match.group(1).upper() if match else None


def _source_registration(source_path):
    if source_path is None:
        return None, None
    # A missing/unreadable explicitly supplied source is an input error.
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    matches = [row for row in _catalog()["bindings"].values()
               if row["source_sha256"] == source_hash]
    if len(matches) > 1:
        raise ValueError("ambiguous_simulator_source")
    return (deepcopy(matches[0]) if matches else None), source_hash


def expected_revision(scenario, *, source_path=None):
    """Resolve the same mechanism identity as the factory, without model calls.

    A matching judge scenario key does not establish simulator compatibility:
    resume identity includes the exact source bytes and the prompt profile.
    """
    is_aim = bool(scenario is not None and scenario.state_queue.is_aim_driven())
    if not (is_aim and _task(scenario) in _TASKS and not scenario.composite_type):
        return "legacy"
    from benchmark.user.entry_profile import select_registration, revision as entry_revision
    _, source_hash = _source_registration(source_path)
    entry, _ = select_registration(scenario, source_hash)
    return entry_revision if entry else revision


class _ReleaseAimSimulator(AimDrivenUserSimulator):
    """Keep runtime bindings across resets without changing source objects."""

    def __init__(self, scenario, *, registration, source_hash, candidate_options=None, **kwargs):
        self._release_registration = deepcopy(registration)
        self._release_source_hash = source_hash
        self._candidate_options = deepcopy(candidate_options or {})
        super().__init__(scenario=scenario, **kwargs)

    def set_scenario(self, scenario):
        self._release_source_template = deepcopy(scenario)
        from benchmark.user.minimal_source_candidate import MinimalSourceCandidate
        self._source_candidate = MinimalSourceCandidate(
            scenario, source_hash=self._release_source_hash, **self._candidate_options)
        self.candidate_profile = self._source_candidate.profile
        self.candidate_events = self._source_candidate.events
        from benchmark.user.entry_profile import EntryRenderer, select_registration, revision as entry_revision
        entry, entry_reason = select_registration(scenario, self._release_source_hash)
        self._entry_renderer = EntryRenderer(scenario, entry) if entry else None
        self.prompt_interventions = self._entry_renderer.events if entry else []
        self.entry_profile = dict(
            revision=entry_revision, status="configured" if entry else "unconfigured",
            reason=entry_reason, source_sha256=self._release_source_hash,
            scenario_key=entry["scenario_key"] if entry else None,
            opening="O1" if entry and entry.get("opening") else None,
            generation=("/".join(variant for active, variant in (
                (entry["generation"].get("preference_aim_id"), "R2"),
                (entry["generation"].get("material_aim_ids"), "R2"),
                (entry["generation"].get("final_aim_id"), "O3")) if active)
                if entry and entry.get("generation") else None),
            stage_bindings=deepcopy(entry.get("generation")) if entry else None,
            evidence_scope=entry["evidence_scope"] if entry else None)
        private = deepcopy(scenario)
        # Production selection is controlled by this catalog, not arbitrary
        # metadata on an unregistered input document.
        if private.metadata and "simulator_no_repeat" in private.metadata:
            private.metadata.pop("simulator_no_repeat")
        registration = self._release_registration
        matches = bool(registration and
                       _digest_scenario(scenario) == registration["scenario_sha256"])
        if matches:
            private.metadata = {**(private.metadata or {}),
                                "simulator_no_repeat": deepcopy(registration["binding"])}
        self.release_profile = dict(
            revision=revision, variant="V3", generation="B1",
            preference_status="configured" if matches else "unconfigured",
            preference_reason=("registered_source" if matches else
                "scenario_object_mismatch" if registration else "source_not_registered"),
            scenario_key=registration["scenario_key"] if matches else None,
            source_sha256=self._release_source_hash,
            content_repair=False, numeric_note="not_adopted")
        if entry:
            self.release_profile.update(revision=entry_revision, base_revision=revision,
                                        prompt_profile=deepcopy(self.entry_profile))
        super().set_scenario(private)
        if hasattr(self, '_identity_options'):
            from benchmark.user.simulator_identity import build_simulator_identity
            self.__dict__.update(build_simulator_identity(scenario, **self._identity_options))

    def _render_opening_messages(self, messages, *, anchor_guidance, is_disguised):
        if self._entry_renderer is None:
            return messages
        return self._entry_renderer.opening(
            messages, anchor_guidance=anchor_guidance, is_disguised=is_disguised)

    def _render_generation_messages(self, messages, *, aim, history):
        baseline = (messages if self._entry_renderer is None else
                    self._entry_renderer.generation(messages, aim=aim, history=history))
        return self._source_candidate.generation(baseline, aim=aim, history=history)

    def reset(self):
        if self.scenario:
            self.set_scenario(self._release_source_template)
        else:
            super().reset()


def create_release_simulator(scenario, model, *, source_path=None, trace=None,
                             llm_args=None, perception_llm_args=None,
                             generation_candidate=None, candidate_source_key=None,
                             candidate_bindings=(), candidate_strict=False, **kwargs):
    """Create the release simulator; pass a budget through ``trace.budget``.

    Native, aim-driven T1--T5 uses frozen B1/V3 with optional registered P1.
    Other task types retain the legacy simulator. Model and provider options
    do not select or modify the mechanism. No completion is made here.
    """
    from benchmark.user.simulator_identity import build_simulator_identity
    identity_state = build_simulator_identity(scenario, source_path=source_path,
        generation_candidate=generation_candidate, candidate_source_key=candidate_source_key,
        candidate_bindings=candidate_bindings, candidate_strict=candidate_strict)
    candidate_source_key = identity_state['simulator_identity']['source_key']
    is_aim = bool(scenario is not None and scenario.state_queue.is_aim_driven())
    scoped = bool(is_aim and _task(scenario) in _TASKS and not scenario.composite_type)
    if not scoped:
        cls = AimDrivenUserSimulator if is_aim else UserSimulator
        options = dict(kwargs)
        if is_aim:
            options.update(trace=trace, perception_llm_args=deepcopy(perception_llm_args))
        sim = cls(scenario=scenario, llm=model, llm_args=deepcopy(llm_args), **options)
        sim.release_profile = dict(revision="legacy", variant="V0",
                                   preference_status="unconfigured")
        sim.candidate_profile = dict(name=generation_candidate, status="unconfigured",
                                     reason="not_applicable")
        sim.candidate_events = []
        sim.__dict__.update(identity_state)
        return sim
    if "experiment_variant" in kwargs or "task_controller" in kwargs:
        raise ValueError("release_simulator_mechanism_override_not_supported")
    trace = trace if trace is not None else FirstPassTrace(content_repair=False)
    if trace.content_repair:
        raise ValueError("release_simulator_requires_content_repair_off")
    if trace.budget is None:
        # Shared runner budget: opening, P + U per aim, and N + 1 agent replies.
        trace.budget = CallBudget(limit=2 + 3 * len(scenario.state_queue.aim_sequence.aims))
    trace.budget.reject_invalid_responses = True
    generation_options = {**deepcopy(llm_args or {}), "max_retries": 0, "num_retries": 0}
    perception_options = {**deepcopy(perception_llm_args or {}), "max_retries": 0, "num_retries": 0}
    registration, source_hash = _source_registration(source_path)
    sim = _ReleaseAimSimulator(scenario, registration=registration,
        candidate_options=dict(name=generation_candidate, source_key=candidate_source_key,
                               bindings=candidate_bindings, strict=candidate_strict),
        source_hash=source_hash, llm=model, llm_args=generation_options, trace=trace,
        perception_llm_args=perception_options, experiment_variant="V3", **kwargs)

    sim.__dict__.update(identity_state)
    sim._identity_options = dict(source_path=source_path, generation_candidate=generation_candidate,
        candidate_source_key=candidate_source_key, candidate_bindings=deepcopy(candidate_bindings),
        candidate_strict=candidate_strict)
    return sim
