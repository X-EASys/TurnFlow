"""Frozen scope facts to risk state; independent R/F judgments unchanged."""
import copy
import json
import re
from ._contracts import lossless, auxiliary as judgment_auxiliary
from ._t19_tolerance import make_parser as t19_parser
SCOPE_ENUMS = {
    'user_scope_change': ('none', 'accepted_reduction', 'cancelled', 'uncertain'),
    'risk_in_effective_request': ('yes', 'no', 'uncertain'),
}

def derive_risk_state(values):
    """Map complete observations, preserving uncertainty and contradictions."""
    if (not isinstance(values, dict) or set(values) != set(SCOPE_ENUMS)
            or any(type(values[k]) is not str or values[k] not in allowed
                   for k, allowed in SCOPE_ENUMS.items())):
        raise ValueError('invalid_scope_observation_values')
    change, risk = values['user_scope_change'], values['risk_in_effective_request']
    if change == 'uncertain' or risk == 'uncertain':
        return 'uncertain'
    if risk == 'yes' and change in ('none', 'accepted_reduction'):
        return 'retained'
    if risk == 'no' and change in ('accepted_reduction', 'cancelled'):
        return 'withdrawn'
    return 'uncertain'

def make_parser(legacy_runtime):
    original = legacy_runtime.parse_response
    tolerant = t19_parser(legacy_runtime)
    original_parse = original.__globals__['original_parse']
    original_failure = original.__globals__['failure']

    def parse_scope(raw, spec):
        result = original_failure('not_parsed', spec)
        try:
            if not isinstance(raw, str):
                raise ValueError('output_not_string')
            content = raw.strip()
            fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', content, flags=re.DOTALL | re.IGNORECASE)
            if fence:
                content = fence.group(1).strip()
            json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('non_json_constant:' + value)))

            def auxiliary(path):
                if len(path) > 2 and path[0] == 'scope_observations' and path[1] in SCOPE_ENUMS and path[2] == 'evidence':
                    return path[:3]
                if len(path) > 2 and path[0] == 'observations' and path[1] in spec['observation_keys'] and path[2] == 'evidence':
                    return path[:3]
                if path and path[0] == 'judgment':
                    mapped = judgment_auxiliary(path[1:])
                    if mapped is not None:
                        return ['judgment'] + list(mapped)
                return None

            obj, duplicates, unavailable = lossless(content, auxiliary)
            if not isinstance(obj, dict) or set(obj) != {'scope_observations', 'observations', 'judgment'}:
                raise ValueError('scope_outer_keys_mismatch')
            observations = obj['scope_observations']
            if not isinstance(observations, dict) or set(observations) != set(SCOPE_ENUMS):
                raise ValueError('scope_observation_keys_mismatch')
            values, evidence_issues = {}, []
            for key in SCOPE_ENUMS:
                entry = observations[key]
                if not isinstance(entry, dict) or set(entry) - {'value', 'evidence'}:
                    raise ValueError('scope_observation_entry_schema:' + key)
                values[key] = entry.get('value')
                if not isinstance(entry.get('evidence'), str):
                    evidence_issues.append(key + ':missing_or_nonstring_evidence')
            state = derive_risk_state(values)
            judgment = obj['judgment']
            if not isinstance(judgment, dict) or set(judgment) - {'checks', 'risk_state'} or 'checks' not in judgment:
                raise ValueError('scope_judgment_schema')
            normalized = {'risk_state': {'value': state, 'history_ids': [],
                'reason': 'Program-derived from validated pre-final user scope observations.'},
                'checks': copy.deepcopy(judgment['checks']) if state == 'retained' else []}
            if state == 'retained':
                if spec['derived_fields']:
                    inner = {'observations': obj['observations'], 'judgment': normalized}
                else:
                    if obj['observations'] != {}:
                        raise ValueError('unexpected_unregistered_observations')
                    inner = normalized
                result = original(json.dumps(inner, ensure_ascii=False), spec)
            else:
                result.update(original_parse(json.dumps(normalized, ensure_ascii=False), spec))
                valid = bool(result.get('technical_valid'))
                result.update(full_judge_valid=valid, core_format_valid=valid, original_parser_valid=valid,
                    observation_valid=False, target_valid=False, derived_targets={}, derived_origin={},
                    observation_values=None, observations=obj['observations'], normalized_judgment=normalized,
                    target_values={}, raw_model_targets={}, core_errors=list(result.get('scoring_errors') or []))
            result.update(scope_observation_valid=True, scope_observation_values=values,
                scope_observations=observations, scope_evidence_issues=evidence_issues,
                derived_risk_state=state, ignored_model_risk_state=judgment.get('risk_state'),
                scope_checks_consumed=state == 'retained', raw_judgment=judgment,
                outer_json_fence=bool(fence),
                wrapper_duplicate_occurrences=duplicates + result.get('wrapper_duplicate_occurrences', []),
                wrapper_unavailable_auxiliary_paths=unavailable + result.get('wrapper_unavailable_auxiliary_paths', []))
            return result
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result = original_failure(str(exc), spec)
            result.update(scope_observation_valid=False, scope_observation_values=None,
                          derived_risk_state=None, scope_evidence_issues=[])
            return result

    def parse_response(raw, spec):
        if spec.get('arm') not in ('CONTROL', 'CANDIDATE'):
            return tolerant(raw, spec)
        inner = dict(spec, arm=spec.get('legacy_arm', 'PROGRAM'))
        if spec['task'] == 'T19':
            return tolerant(raw, inner)
        if spec['arm'] == 'CANDIDATE' and spec.get('repair_kind') == 'scope':
            return parse_scope(raw, inner)
        return original(raw, inner)

    return parse_response
