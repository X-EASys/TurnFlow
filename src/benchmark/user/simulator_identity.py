"""Path-independent simulator identity shared by planning and the factory."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'simulator-identity-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_candidate_bindings(path):
    if path is None:
        return ()
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data if isinstance(data, list) else data['registrations']
    seen = set()
    for row in rows:
        required = {'source_key', 'source_sha256', 'scenario_sha256', 'aim_id',
                    'task', 'arms', 'turn_role', 'speaker', 'dependencies'}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError('invalid_candidate_binding_schema')
        key = (row['source_key'], row['aim_id'])
        if key in seen or row['task'] not in ('T17', 'T20') or not row['source_key'].startswith(row['task'] + '/'):
            raise ValueError('duplicate_or_invalid_candidate_binding')
        if row['arms'] != ['R'] or row['turn_role'] != 'material' or not isinstance(row['dependencies'], list):
            raise ValueError('invalid_candidate_binding_role')
        for field in ('source_sha256', 'scenario_sha256'):
            if not isinstance(row[field], str) or len(row[field]) != 64:
                raise ValueError('invalid_candidate_binding_hash')
        for dependency in row['dependencies']:
            if not {'aim_id', 'include_intent', 'payload_keys'} <= dependency.keys():
                raise ValueError('invalid_candidate_dependency')
        seen.add(key)
    return deepcopy(rows)


def source_key_for(scenario, source_path):
    from benchmark.user.release_profile import _task
    task = _task(scenario)
    try:
        from benchmark.evaluator.task_entry import load_source
        return load_source(task, source_path)[0]
    except (OSError, ValueError, KeyError):
        source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else digest(scenario.model_dump(mode='json'))
        return f'{task or "legacy"}/unregistered/{source_hash}'


def build_simulator_identity(scenario, *, source_path=None, generation_candidate=None,
                             candidate_source_key=None, candidate_bindings=(),
                             candidate_strict=False, kind='bridge', dry_run=False):
    from benchmark.user.release_profile import expected_revision
    from benchmark.user.minimal_source_candidate import MinimalSourceCandidate
    from benchmark.user.perception import PERCEPTION_PARSER_POLICY
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else None
    authoritative_key = source_key_for(scenario, source_path)
    if candidate_source_key and '/unregistered/' not in authoritative_key and candidate_source_key != authoritative_key:
        raise ValueError('candidate_source_identity_mismatch')
    key = candidate_source_key or authoritative_key
    applicable = kind == 'bridge' and not dry_run
    candidate = MinimalSourceCandidate(scenario, name=generation_candidate if applicable else None,
        source_key=key, source_hash=source_hash, bindings=candidate_bindings,
        strict=candidate_strict)
    profile = deepcopy(candidate.profile)
    if not applicable:
        profile.update(name=generation_candidate, status='unconfigured' if generation_candidate else 'disabled',
                       reason='not_applicable')
    active = applicable and candidate.active
    fields = ('task', 'source_key', 'source_sha256', 'scenario_sha256', 'aim_id',
              'turn_role', 'speaker', 'arms', 'dependencies')
    effective = [{k: deepcopy(r[k]) for k in fields} for r in candidate_bindings if active and r['source_key'] == key
                 and r['source_sha256'] == source_hash
                 and r['scenario_sha256'] == digest(scenario.model_dump(mode='json'))]
    # Dependencies are a set of source references; registration ordering and
    # unrelated source entries cannot invalidate an otherwise identical case.
    for row in effective:
        row['dependencies'] = sorted(row['dependencies'], key=digest)
    effective.sort(key=digest)
    paths = ['release_profile.py', 'aim_driven_simulator.py', 'user_simulator.py',
             'entry_profile.py', 'resources/release_profile.json',
             'resources/simulator_entry_profile.json', 'bridge_generator.py', 'prompts.py',
             'perception.py', 'translate.py', 'policy.py', 'task_specific_controls.py',
             'typed_generation_boundary.py', 'opening_prompt_options.py',
             'preference_midturn_options.py', 'preference_prompt_options.py'] if applicable else []
    if active:
        paths += ['minimal_context_repair.py', 'task_specific_repair.py', 'minimal_source_candidate.py']
    directory = Path(__file__).parent
    templates = {p: hashlib.sha256((directory / p).read_bytes()).hexdigest() for p in paths}
    identity = dict(schema=SCHEMA, mechanism=expected_revision(scenario, source_path=source_path) if applicable else kind,
        kind=kind, dry_run=dry_run, candidate='minimal_source_r_v1' if active else None,
        candidate_version='v1' if active else None, source_key=key,
        source_bytes_sha256=source_hash, parsed_scenario_sha256=digest(scenario.model_dump(mode='json')),
        template_fingerprint=digest(templates), template_files=templates,
        effective_bindings_sha256=digest(effective))
    if applicable:
        identity['perception_parser_policy'] = PERCEPTION_PARSER_POLICY
    return dict(simulator_identity=identity,
        candidate_configuration=dict(requested=dict(name=generation_candidate, strict=bool(candidate_strict)),
                                     effective=profile))


def simulator_state(simulator):
    return dict(simulator_identity=deepcopy(getattr(simulator, 'simulator_identity', None)),
        candidate_configuration=deepcopy(getattr(simulator, 'candidate_configuration', None)),
        candidate_profile=deepcopy(getattr(simulator, 'candidate_profile', {})),
        candidate_events=deepcopy(getattr(simulator, 'candidate_events', [])),
        material_activation_count=sum(e.get('activated', False) and e.get('turn_role') == 'material'
                                      for e in getattr(simulator, 'candidate_events', [])))
