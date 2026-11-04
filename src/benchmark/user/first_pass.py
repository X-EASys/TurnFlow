"""Explicit, per-run evidence collection for simulator experiments."""
from dataclasses import dataclass, field
from copy import deepcopy
from time import monotonic, sleep


@dataclass
class CallBudget:
    limit: int
    used: int = 0
    min_interval_seconds: float = 0.0
    last_attempt_at: float = 0.0

    def consume(self):
        if self.used >= self.limit:
            raise RuntimeError("First-pass API attempt budget exhausted")
        delay = self.min_interval_seconds - (monotonic() - self.last_attempt_at)
        if delay > 0:
            sleep(delay)
        self.last_attempt_at = monotonic()
        self.used += 1


@dataclass
class FirstPassTrace:
    content_repair: bool = False
    budget: object = None
    stages: list = field(default_factory=list)
    content_changes: list = field(default_factory=list)
    execution_errors: list = field(default_factory=list)
    perception_replay: dict | None = None

    def change(self, name, before, after):
        if before != after:
            self.content_changes.append(dict(name=name, before=deepcopy(before), after=deepcopy(after)))
        return after


def call_stage(trace, name, fn, **kwargs):
    if trace is None:
        return fn(**kwargs)
    stage = dict(name=name, prompt=deepcopy(kwargs.get('messages', [])),
                 model_args={k: kwargs[k] for k in ('model', 'temperature', 'max_tokens') if k in kwargs},
                 raw_response=None, error=None, attempts=0, tokens=None)
    trace.stages.append(stage)
    start = monotonic()
    try:
        if name == "PERCEIVE" and trace.perception_replay is not None:
            import hashlib
            import json
            replay = trace.perception_replay
            actual = hashlib.sha256(json.dumps(stage['prompt'], sort_keys=True,
                ensure_ascii=False).encode()).hexdigest()
            if actual != replay['prompt_sha256']:
                raise ValueError('Frozen PERCEIVE prompt mismatch')
            stage.update(replayed=True, replay_source=replay['source'],
                         replay_sha256=replay['raw_sha256'], parser_revision='intent-v1')
            result = replay['raw_response']
            if hashlib.sha256(result.encode()).hexdigest() != replay['raw_sha256']:
                raise ValueError('Frozen PERCEIVE response mismatch')
            stage['raw_response'] = result
            return result
        result = fn(_first_pass_stage=stage, _first_pass_budget=trace.budget, **kwargs)
        stage['raw_response'] = result
        return result
    except BaseException as exc:
        # Exception types are sufficient for classification; provider errors can contain credentials.
        stage['error'] = type(exc).__name__
        raise
    finally:
        stage['latency_seconds'] = monotonic() - start
        if hasattr(trace.budget, 'record_stage'):
            trace.budget.record_stage(stage)


def repair(collector, name, fn, content, *args, **kwargs):
    if collector is not None and not collector.content_repair:
        return content
    before = deepcopy(content)
    result = fn(content, *args, **kwargs)
    return collector.change(name, before, result) if collector is not None else result


def restore_checkpoint(simulator, checkpoint):
    """Restore the finite aim-driven state without replaying model calls.

    Later checkpoints currently support the controller-free T1–T5 route;
    unsupported or underspecified state fails rather than guessing disclosure.
    """
    from benchmark.data_model.aim import AimStatus
    from benchmark.data_model.message import UserMessage, AssistantMessage
    from benchmark.user.base import UserContext
    history = []
    for item in checkpoint.get('history', []):
        cls = {'user': UserMessage, 'assistant': AssistantMessage}.get(item['role'])
        if cls is None:
            raise ValueError('Checkpoint may contain only user/assistant messages')
        history.append(cls(content=item['content'], metadata=deepcopy(item.get('metadata')),
                           turn_idx=item.get('turn_idx')))
    index = checkpoint['aim_index']
    state = checkpoint.get('state')
    if index or history:
        if state is None or simulator.task_controller is not None:
            raise ValueError('Noninitial checkpoint requires explicit controller-free runtime state')
        sequence = simulator.aim_sequence
        if len(state['aims']) != len(sequence.aims):
            raise ValueError('Checkpoint aim state does not match scenario')
        simulator.context = UserContext.model_validate(state['context'])
        sequence.current_aim_index = index
        sequence.total_turns_spent = state['total_aim_turns']
        for aim, saved in zip(sequence.aims, state['aims']):
            if aim.aim_id != saved['aim_id']:
                raise ValueError('Checkpoint aim identity mismatch')
            aim.status = AimStatus(saved['status'])
            aim.turns_spent = saved['turns_spent']
            aim.info_disclosed = list(saved['info_disclosed'])
            aim.request_made = saved['request_made']
        simulator.aim_state_tracker.started_aims = set(state['started_aims'])
        simulator.aim_state_tracker.aim_entry_turns = dict(state['aim_entry_turns'])
        hidden = state['hidden_disclosed']
        if set(hidden) != {h.key for h in simulator._hidden_state}:
            raise ValueError('Checkpoint hidden-state identity mismatch')
        for item in simulator._hidden_state:
            item.disclosed_during_conversation = hidden[item.key]
        simulator._anchor_revealed = state['anchor_revealed']
        # Aim-driven step does not update the legacy tracker. Require its
        # explicitly documented reset state for this checkpoint format.
        if state['legacy_tracker'] != 'reset':
            raise ValueError('Unsupported legacy tracker checkpoint')
    current = simulator.aim_sequence.get_current_aim()
    if current is None or current.aim_id != checkpoint['aim_id']:
        raise ValueError('Restored aim does not match checkpoint')
    return history
