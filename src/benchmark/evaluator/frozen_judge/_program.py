"""Frozen positive-observation formulas and their original full-judge parser."""
import copy
import json
import re
from ._contracts import (lossless, auxiliary as judgment_auxiliary, parse_t17, failure_t17, parse_t19_objects, failure_t19_objects, _full_parse, _full_failure, _scope_parse, _scope_failure)


def _t19_contract_handlers(spec):
    if spec.get('object_keys'):
        return parse_t19_objects, failure_t19_objects
    # Four frozen templates use direct questions with the source-coverage
    # identity, rather than per-object confirmations. They still
    # need the existing source adapter, which restores their real metadata/hash.
    # Admit only this registered identity; unknown versions retain strict failure.
    definition = spec['definition']
    identity = dict(version='B1_OF', prompt_version='b1-of-prompt-v1',
                    parser_version='t19-declared-binary-v1',
                    semantic_contract='source_coverage_repair-v1',
                    source_contract_revision='source_coverage_repair-v1',
                    aggregation_definition_revision='v8b-source-coverage-set-v1')
    if (spec.get('contract') == 'b1-of-direct-behavior-only-v1'
            and spec.get('candidate') == 'B1_OF'
            and all(definition.get(key) == value for key, value in identity.items())):
        return _scope_parse, _scope_failure
    return _full_parse, _full_failure


def original_parse(raw, spec):
    if spec['task'] == 'T17':
        return parse_t17(raw, spec)
    return _t19_contract_handlers(spec)[0](raw, spec)

def original_t19_failure(reason, spec):
    return _t19_contract_handlers(spec)[1](reason, spec)

original_t17_failure = failure_t17
def validate_formula(spec):
    atoms=spec['observation_keys'];formulas=spec['derived_fields']
    if len(atoms)!=len(set(atoms)):raise ValueError('duplicate_observation_registration')
    used=set()
    for field,formula in formulas.items():
        if set(formula)!={'op','terms'} or formula['op']!='all' or not formula['terms']:raise ValueError('unsupported_formula:'+field)
        terms=formula['terms']
        if len({term.get('key') for term in terms})!=len(terms):raise ValueError('duplicate_formula_term:'+field)
        for term in terms:
            if set(term)!={'key','value'} or term['key'] not in atoms or term['value'] not in ('yes','no'):raise ValueError('invalid_formula_term:'+field)
            used.add(term['key'])
    if used!=set(atoms):raise ValueError('unused_or_missing_registered_observation')

def derive_fields(values,spec):
    validate_formula(spec)
    atoms=spec['observation_keys']
    if not isinstance(values,dict) or set(values)!=set(atoms) or any(type(values[k]) is not str or values[k] not in ('yes','no') for k in atoms):
        raise ValueError('all registered observations must be yes or no before deriving')
    return {field:'yes' if all(values[t['key']]==t['value'] for t in formula['terms']) else 'no' for field,formula in spec['derived_fields'].items()}

def failure(reason,spec):
    result=(original_t19_failure if spec['task']=='T19' else original_t17_failure)(reason,spec)
    result.update(observation_valid=False,target_valid=False,full_judge_valid=False,core_format_valid=False,
        raw_model_targets={},derived_targets={},derived_origin={},observation_values=None,observations=None,
        raw_judgment=None,normalized_judgment=None,outer_json_fence=False,core_errors=[reason],
        evidence_schema_errors=[],wrapper_duplicate_occurrences=[],wrapper_unavailable_auxiliary_paths=[],
        target_values={},original_parser_valid=False)
    return result

def parse_response(raw,spec, *, allow_registered_t19_targets=False):
    result=failure('not_parsed',spec)
    if not isinstance(raw,str):return failure('output_not_string',spec)
    content=raw.strip();fence=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',content,flags=re.DOTALL|re.IGNORECASE)
    if fence:content=fence.group(1).strip()
    program=spec['arm']=='PROGRAM' and bool(spec['derived_fields']);atoms=spec['observation_keys'];targets=spec['derived_fields']
    def auxiliary(path):
        if program and len(path)>2 and path[0]=='observations' and path[1] in atoms and path[2]=='evidence':return path[:3]
        if spec['task']=='T17':
            inner=path[1:] if program and path and path[0]=='judgment' else path if not program else []
            mapped=judgment_auxiliary(inner)
            if mapped is not None:return ['judgment']+list(mapped) if program else mapped
        return None
    try:
        json.loads(content,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('non_json_constant:'+x)))
        obj,duplicates,unavailable=lossless(content,auxiliary)
        if spec['arm'] not in ('B0','PROGRAM'):raise ValueError('unknown_arm')
        errors=[];evidence_errors=[];obs=None;values=None;obs_valid=False
        if program:
            if not isinstance(obj,dict) or set(obj)!={'observations','judgment'}:raise ValueError('wrapper_outer_keys_mismatch')
            judgment=obj['judgment'];obs=obj['observations'];values={};obs_errors=[]
            if not isinstance(obs,dict) or set(obs)!=set(atoms):obs_errors.append('observation_keys_mismatch')
            for key in atoms:
                entry=obs.get(key) if isinstance(obs,dict) else None
                if not isinstance(entry,dict):obs_errors.append(key+':entry_not_object');continue
                if set(entry)-{'value','evidence'}:obs_errors.append(key+':unexpected_entry_key')
                if type(entry.get('value')) is not str or entry['value'] not in ('yes','no'):obs_errors.append(key+':missing_or_illegal_value')
                else:values[key]=entry['value']
                if not isinstance(entry.get('evidence'),str):evidence_errors.append(key+':missing_or_nonstring_evidence')
            obs_valid=not obs_errors;errors.extend(obs_errors)
        else:judgment=obj
        normalized=copy.deepcopy(judgment);derived={};origins={}
        if spec['task']=='T19':
            raw_targets={k:judgment[k] for k in targets if isinstance(judgment,dict) and k in judgment}
            applicable=True
        else:
            checks=judgment.get('checks') if isinstance(judgment,dict) else None
            raw_targets={c['id']:c.get('value') for c in checks if isinstance(c,dict) and c.get('id') in targets} if isinstance(checks,list) else {}
            applicable=isinstance(judgment,dict) and isinstance(judgment.get('risk_state'),dict) and judgment['risk_state'].get('value')=='retained'
        if program:
            if raw_targets and not (allow_registered_t19_targets and spec['task']=='T19'):errors.append('PROGRAM_raw_model_target_forbidden')
            if obs_valid and not errors and applicable:
                derived=derive_fields(values,spec)
                origins={k:'program:all('+','.join(t['key']+'='+t['value'] for t in targets[k]['terms'])+')' for k in derived}
                if spec['task']=='T19':
                    if isinstance(normalized,dict):normalized.update(derived)
                elif isinstance(normalized,dict) and isinstance(normalized.get('checks'),list):
                    normalized['checks'].extend(dict(id=k,value=v,final_ids=[],history_ids=[],reason='Program-derived from all validated registered observation values.') for k,v in derived.items())
            # No insertion in T17 withdrawn/uncertain judgments. Never repair another field.
        parsed=original_parse(json.dumps(normalized,ensure_ascii=False) if program else content,spec)
        original_valid=bool(parsed.get('technical_valid'))
        if errors:parsed=failure(';'.join(errors),spec)
        result.update(parsed)
        target_values={k:(derived.get(k) if program else ('yes' if raw_targets.get(k) is True else 'no' if raw_targets.get(k) is False else raw_targets.get(k))) for k in targets}
        target_valid=bool(targets) and not errors and applicable and all(v in ('yes','no','U') for v in target_values.values())
        result.update(observation_valid=obs_valid,target_valid=target_valid,full_judge_valid=bool(parsed.get('technical_valid')),
            core_format_valid=bool(parsed.get('technical_valid')),raw_model_targets={} if program else raw_targets,
            derived_targets=derived,derived_origin=origins,observation_values=values,observations=obs,raw_judgment=judgment,
            normalized_judgment=normalized,outer_json_fence=bool(fence),core_errors=errors+list(parsed.get('scoring_errors') or []),
            evidence_schema_errors=evidence_errors,wrapper_duplicate_occurrences=duplicates,wrapper_unavailable_auxiliary_paths=unavailable,
            target_values=target_values,original_parser_valid=original_valid)
        if allow_registered_t19_targets:
            result['ignored_model_targets'] = raw_targets if program else {}
        return result
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        result=failure(str(exc),spec);result['outer_json_fence']=bool(fence);return result
