"""Frozen scope parsing policy, promoted from the separately audited replay.

Historical experiment gates remain unchanged; the public adapter records the
current policy revision on outputs that use this tolerance.

Ignore wholly unconsumed observation text or registered program-derived target
values, then reuse the complete frozen parser for all consumed scope/observation/
R/F values. Never inserts reference values or corrects independent judgments.
"""
import copy
import json

from ._contracts import lossless
from ._contracts import auxiliary as judgment_auxiliary


def make_parser(base_parser):
    """Wrap an existing ``parse_response(raw, spec)`` without mutating it."""
    def parse_response(raw, spec):
        original = base_parser(raw, spec)
        if (original.get('technical_valid')
                or spec.get('task') != 'T17'
                or spec.get('arm') != 'CANDIDATE'
                or spec.get('repair_kind') != 'scope'):
            return original
        errors = original.get('core_errors') or []
        empty_observations = (spec.get('observation_keys') == [] and spec.get('derived_fields') == {}
            and 'unexpected_unregistered_observations' in errors)
        registered_targets = (isinstance(spec.get('derived_fields'), dict) and bool(spec['derived_fields'])
            and bool(spec.get('observation_keys'))
            and 'PROGRAM_raw_model_target_forbidden' in errors)
        if not (empty_observations or registered_targets):
            return original

        def auxiliary(path):
            if (len(path) > 2 and path[0] == 'scope_observations'
                    and path[1] in ('user_scope_change', 'risk_in_effective_request')
                    and path[2] == 'evidence'):
                return path[:3]
            if (len(path) > 2 and path[0] == 'observations'
                    and path[1] in spec.get('observation_keys', []) and path[2] == 'evidence'):
                return path[:3]
            if path and path[0] == 'judgment':
                mapped = judgment_auxiliary(path[1:])
                if mapped is not None:
                    return ['judgment'] + list(mapped)
            return None

        try:
            obj, duplicates, unavailable = lossless(raw, auxiliary)
            if not isinstance(obj, dict):
                return original
            normalized = copy.deepcopy(obj)
            ignored, ignored_targets = None, None
            if empty_observations:
                if not isinstance(obj.get('observations'), dict) or not obj['observations']:
                    return original
                ignored = normalized['observations']
                normalized['observations'] = {}
            else:
                checks = normalized.get('judgment', {}).get('checks')
                if not isinstance(checks, list):
                    return original
                seen = set()
                for check in checks:
                    identifier = check.get('id') if isinstance(check, dict) else None
                    if isinstance(identifier, str):
                        if identifier in seen:
                            return original
                        seen.add(identifier)
                targets = set(spec['derived_fields'])
                ignored_targets = [check for check in checks if isinstance(check, dict) and check.get('id') in targets]
                if not ignored_targets:
                    return original
                normalized['judgment']['checks'] = [check for check in checks if not isinstance(check, dict) or check.get('id') not in targets]
            result = base_parser(json.dumps(normalized, ensure_ascii=False), spec)
            if ignored is not None:
                result['ignored_scope_observations'] = ignored
            if ignored_targets is not None:
                result['ignored_scope_derived_targets'] = ignored_targets
            result['posthoc_auxiliary_tolerance'] = {
                'not_formal_result': True,
                'policy': ('Ignore observations only when no observation or derived field is registered; revalidate all consumed core values.'
                    if empty_observations else 'Ignore only registered model-produced derived targets; validate original observations and all remaining core values before program derivation.'),
                'original_raw_output': raw,
                'original_core_errors': copy.deepcopy(original.get('core_errors')),
                'original_technical_valid': original.get('technical_valid'),
                'normalized_payload': normalized,
            }
            result['wrapper_duplicate_occurrences'] = duplicates + result.get('wrapper_duplicate_occurrences', [])
            result['wrapper_unavailable_auxiliary_paths'] = unavailable + result.get('wrapper_unavailable_auxiliary_paths', [])
            return result
        except (ValueError, TypeError, KeyError, AttributeError):
            return original

    return parse_response
