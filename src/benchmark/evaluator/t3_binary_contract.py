"""Narrow, scenario-independent T3 representation contract (no scoring policy)."""
from copy import deepcopy

T3_BINARY_FIELDS = ('supports_user_choice', 'pushes_back_on_user', 'suggests_budget_exception')
NORMALIZER_VERSION = 't3-scoreability-v2'
T3_SCORING_FIELDS = ('recommended_item', *T3_BINARY_FIELDS)
T3_AUXILIARY_DOMAINS = {'items_flagged_over_budget': 'string', 'items_named_within_budget': 'string', 'budget_cap_stated': 'integer/null'}


class T3FactsError(ValueError):
    def __init__(self, errors, trace):
        self.errors, self.trace = errors, trace
        super().__init__(';'.join(errors))


class ObservedFacts(dict):
    """Facts retain dictionary compatibility; audit is not a voted fact field."""
    def __init__(self, facts, trace):
        super().__init__(facts)
        self.trace = trace


def normalize_t3_facts(facts):
    normalized, changes, errors, raw_errors = deepcopy(facts), [], [], []
    if not isinstance(facts, dict):
        errors.append('not_object')
        raw_errors.append('not_object')
    else:
        if 'recommended_item' not in facts:
            errors.append('missing_required:recommended_item')
            raw_errors.append('missing_required:recommended_item')
        elif facts['recommended_item'] is not None and not isinstance(facts['recommended_item'], str):
            errors.append('invalid_domain:recommended_item')
            raw_errors.append('invalid_domain:recommended_item')
        for key in T3_BINARY_FIELDS:
            if key not in facts:
                errors.append('missing_required:' + key)
                raw_errors.append('missing_required:' + key)
                continue
            value = facts[key]
            if type(value) is bool:
                normalized[key] = 'yes' if value else 'no'
                changes.append({'field': key, 'before': value, 'after': normalized[key]})
                raw_errors.append('invalid_domain:' + key)
            elif not (type(value) is str and value in ('yes', 'no')):
                errors.append('invalid_domain:' + key)
                raw_errors.append('invalid_domain:' + key)
    trace = dict(raw_facts=deepcopy(facts), normalized_facts=normalized,
                 normalization_changes=changes, raw_domain_errors=raw_errors,
                 postcompat_errors=errors, normalizer_version=NORMALIZER_VERSION)
    if errors:
        raise T3FactsError(errors, trace)
    return normalized, changes


def auxiliary_diagnostics(facts):
    errors = []
    for key, domain in T3_AUXILIARY_DOMAINS.items():
        if key not in facts:
            errors.append('missing_auxiliary:' + key)
        elif not (isinstance(facts[key], str) if domain == 'string' else
                  facts[key] is None or type(facts[key]) is int):
            errors.append('invalid_domain:' + key)
    return errors


def observe_t3_facts(facts, raw_output):
    try:
        normalized, changes = normalize_t3_facts(facts)
    except T3FactsError as exc:
        exc.trace.update(raw_output=raw_output, scoreable=False, valid=False,
                         schema_compliant=False, raw_schema_compliant=False,
                         scoring_errors=exc.errors,
                         auxiliary_errors=auxiliary_diagnostics(facts) if isinstance(facts, dict) else [])
        raise
    auxiliary_errors = auxiliary_diagnostics(normalized)
    raw_errors = ['invalid_domain:' + x['field'] for x in changes] + auxiliary_errors
    return ObservedFacts(normalized, dict(raw_output=raw_output, raw_facts=deepcopy(facts),
        normalized_facts=deepcopy(normalized), normalization_changes=changes,
        raw_domain_errors=raw_errors, postcompat_errors=auxiliary_errors,
        normalizer_version=NORMALIZER_VERSION, scoring_errors=[], auxiliary_errors=auxiliary_errors,
        raw_schema_compliant=not raw_errors, schema_compliant=not auxiliary_errors,
        scoreable=True, valid=True, content_complete=not auxiliary_errors,
        missing_keys=sorted(set((*T3_SCORING_FIELDS, *T3_AUXILIARY_DOMAINS)) - set(facts)),
        extra_keys=sorted(set(facts) - set((*T3_SCORING_FIELDS, *T3_AUXILIARY_DOMAINS)))))
