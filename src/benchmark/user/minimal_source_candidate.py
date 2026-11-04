"""Explicit, default-off material routing using the frozen R renderer."""
from copy import deepcopy

from benchmark.user.minimal_context_repair import render
from benchmark.user.task_specific_repair import digest


class MinimalSourceCandidate:
    def __init__(self, scenario, *, name=None, source_key=None, source_hash=None,
                 bindings=(), strict=False):
        if name not in (None, "minimal_source_r_v1"):
            raise ValueError("unknown_simulator_candidate")
        self.scenario = deepcopy(scenario)
        self.events = []
        self.strict = False
        self.source_key = source_key
        self.source_hash = source_hash
        self.bindings = deepcopy(list(bindings))
        from benchmark.user.release_profile import _task
        task = (source_key or "").split("/")[0] or _task(scenario)
        applicable = task in {"T2", "T5"} and not scenario.composite_type
        self.strict = bool(strict and applicable)
        rows = [r for r in self.bindings if r["source_key"] == source_key]
        valid = [r for r in rows if r["source_sha256"] == source_hash
                 and r["scenario_sha256"] == digest(scenario.model_dump(mode="json"))]
        self.active = bool(name and applicable and valid)
        reason = ("disabled" if not name else "not_applicable" if not applicable
                  else "registered_source" if valid else "hash_mismatch" if rows
                  else "source_not_registered")
        self.profile = dict(name=name, status="configured" if self.active else
                            "disabled" if not name else "unconfigured", reason=reason,
                            source_key=source_key, source_sha256=source_hash)
        if name and applicable and strict:
            materials = {a.aim_id for a in scenario.state_queue.aim_sequence.aims
                         if not a.is_final and not a.pre_written_response}
            if (not self.active or len(valid) != len(materials)
                    or {r["aim_id"] for r in valid} != materials
                    or any(r["arms"] != ["R"] or r["turn_role"] != "material"
                           or r["task"] != task for r in valid)):
                raise ValueError("candidate_binding_or_hash_mismatch")

    def generation(self, messages, *, aim, history):
        if not self.profile["name"]:
            return messages
        eligible = not aim.is_final and not aim.pre_written_response
        body = render(self.scenario, source_key=self.source_key or "",
                      source_sha256=self.source_hash, stage="GENERATE", aim=aim,
                      history=history, baseline={"messages": messages},
                      arm="R" if self.active else "B0", registrations=self.bindings)
        activated = body["messages"] != messages
        if self.strict and eligible and not activated:
            raise ValueError("candidate_material_not_activated")
        self.events.append(dict(aim_id=aim.aim_id, turn_role="final" if aim.is_final
                               else "material", activated=activated,
                               status="activated" if activated else
                               "baseline_final" if aim.is_final else "unconfigured",
                               baseline_sha256=digest(messages),
                               actual_sha256=digest(body["messages"])))
        return body["messages"]
