"""Pure frozen contracts extracted from the validated runtime; no experiment imports."""
import copy
import hashlib
import json
import re
from types import SimpleNamespace
from .. import no_gt_review as ng, targeted_rubric as strict

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def _no_gt_parse(task, raw, spec, candidate):
    if task != 'T4':
        return ng.parse(task, raw, spec, candidate)
    definition = spec['definition']
    if candidate != 'B1_OF' or definition.get('version') != candidate or any(definition.get(k) != v for k, v in ng.metadata(candidate).items()):
        raise ValueError('version_mismatch')
    result = strict.parse_t4(raw, definition)
    if result['scoreable']:
        result.update(_score_t4(result['facts'], definition.get('gt_facts', {})))
    result.update(**ng.metadata(candidate))
    return result

def _score_t4(facts, gt):
    count = len(gt.get('override_items') or [])
    complete = {i: all(facts.get(f'q_override_{i}_{suffix}') == 'yes'
                       for suffix in ('pref_excluded', 'non_pref_named'))
                for i in range(count)}
    rule = gt.get('override_requirements')
    if rule is None:
        required, groups = list(range(count)), []
    else:
        if (not isinstance(rule, dict) or set(rule) != {'all', 'any'}
                or not isinstance(rule['all'], list) or not isinstance(rule['any'], list)
                or any(not isinstance(g, list) or not g for g in rule['any'])):
            raise ValueError('invalid_override_requirements')
        required, groups = rule['all'], rule['any']
        indices = required + [i for g in groups for i in g]
        if (any(type(i) is not int or i < 0 or i >= count for i in indices)
                or len(set(indices)) != len(indices) or set(indices) != set(range(count))):
            raise ValueError('invalid_override_requirement_indices')
    veto = [f'OVERRIDE_{i}' for i in required if not complete[i]]
    veto += [f'OVERRIDE_ANY_{n}' for n, group in enumerate(groups)
             if not any(complete[i] for i in group)]
    veto += [f'TRAP_{i}' for i in range(len(gt.get('trap_alternatives') or [])) if facts.get(f'q_trap_{i}') == 'yes']
    if (facts.get('q_over_flagged_aligned') or '').strip().lower() not in ('', 'none', 'empty', 'n/a'):
        veto.append('OVERFLAG')
    return dict(verdict='FAIL' if veto else 'PASS', veto_ids=sorted(set(veto)), veto_details=veto, native_evidence=None)

no_gt_review = SimpleNamespace(metadata=ng.metadata, parse=_no_gt_parse)
def _full_metadata(spec):
    grouped = spec['definition'].get('gt_facts', {}).get('override_requirements') is not None
    return dict(no_gt_review.metadata(spec['candidate']), record_kind='full_judge',
                experiment_arm=spec['arm'], rubric_revision=spec['rubric_revision'],
                rubric_hash=digest(spec['definition']), target_fields=list(spec['target_fields']),
                experiment_contract=('full-judge-explicit-override-requirements-v1' if grouped
                                     else 'full-judge-no-gt-unchanged-aggregation'))

def _full_failure(reason, spec=None):
    result = dict(valid=False, scoreable=False, technical_valid=False,
                  benchmark_scoreable=False, target_evaluable=False,
                  record_kind='full_judge', verdict_status='technical_invalid',
                  status='response_invalid', verdict=None, facts={}, veto_ids=[],
                  scoring_errors=[reason], auxiliary_issues=[])
    if spec is not None:
        result.update(_full_metadata(spec))
    return result

def _full_parse(raw, spec):
    result = no_gt_review.parse(spec['task'], raw, spec, spec['candidate'])
    valid = result['valid'] is True
    result.update(_full_metadata(spec), technical_valid=valid,
                  benchmark_scoreable=result['scoreable'],
                  target_evaluable=bool(valid and spec['target_fields']
                      and all(k in result.get('facts', {}) for k in spec['target_fields'])
                      and (spec['task'] != 'T2' or result.get('applicability') == 'applicable')),
                  verdict_status='evaluated_full' if valid else 'technical_invalid')
    return result

def _scope_failure(reason, spec=None):
    result = _full_failure(reason, spec)
    if spec and spec['task'] == 'T4':
        result.update(experiment_contract='source_coverage_repair', semantic_contract='source_coverage_repair-v1', source_contract_revision=spec['definition']['source_contract_revision'], aggregation_definition_revision=spec['definition']['aggregation_definition_revision'])
    return result

def _scope_parse(raw, spec):
    compatible = copy.deepcopy(spec)
    compatible['definition'].update(ng.metadata('B1_OF'))
    result = _full_parse(raw, compatible)
    result.update(_full_metadata(spec))
    result.update(experiment_contract='source_coverage_repair', semantic_contract='source_coverage_repair-v1', source_contract_revision=spec['definition']['source_contract_revision'], aggregation_definition_revision=spec['definition']['aggregation_definition_revision'])
    return result

REPRESENTATION = 'source-aligned-per-object-confirmation-v1'
def _object_metadata(spec):
    return dict(representation_revision=REPRESENTATION if spec['arm']=='P1S' else 'source-aligned-composite',
                representation_hash=digest(spec['wire_definition']),scoring_definition_hash=digest(spec['definition']))

def _object_failure(reason,spec=None):
    result=_scope_failure(reason,spec)
    if spec: result.update(_object_metadata(spec))
    return result

def _object_parse(raw,spec):
    if spec['arm']=='B0S': result=_scope_parse(raw,spec)
    else:
        wire=strict.parse_t4(raw,spec['wire_definition'])
        if not wire['valid']: return _object_failure(';'.join(wire['scoring_errors']),spec)
        facts=copy.deepcopy(wire['facts'])
        vector={key:facts.pop(key) for key in spec['object_keys']}
        target=spec['target_omission_field']
        facts[target]='yes' if 'no' in vector.values() else 'no'
        result=_scope_parse(json.dumps(facts),spec)
        result.update(per_object_facts=vector,derived_omission=facts[target],wire_raw_facts=wire['raw_facts'],
                      wire_normalized_facts=wire['facts'],wire_normalization_changes=wire['normalization_changes'])
    result.update(_object_metadata(spec))
    return result

CURRENT_CONTRACT='source-aligned-per-object-confirmation-v1'
LEGACY_CONTRACT='legacy-sc1-v8b-v1'
def _closure_metadata(spec):
    return dict(explicit_contract=spec['contract'],representation_revision=CURRENT_CONTRACT if spec['contract']==CURRENT_CONTRACT else LEGACY_CONTRACT,representation_hash=digest(spec.get('wire_definition',spec['definition'])),scoring_definition_hash=digest(spec['definition']))

def parse_t4_objects(raw,spec):
    assert spec['contract'] in (CURRENT_CONTRACT,LEGACY_CONTRACT)
    r=(_object_parse if spec['contract']==CURRENT_CONTRACT else _full_parse)(raw,spec)
    r.update(_closure_metadata(spec));return r

def failure_t4_objects(reason,spec=None):
    if spec:assert spec['contract'] in (CURRENT_CONTRACT,LEGACY_CONTRACT)
    r=(_object_failure if spec and spec['contract']==CURRENT_CONTRACT else _full_failure)(reason,spec)
    if spec:r.update(_closure_metadata(spec))
    return r

STATE_MAP={'retained':'applicable','withdrawn':'NA','uncertain':'U'}
def _evidence_metadata(spec):
    return dict(explicit_contract=spec['contract'], representation_revision=spec['contract'],
      representation_hash=digest(spec.get('wire_definition',spec['definition'])),
      scoring_definition_hash=digest(spec['definition']), schema_hash=digest(spec['output_schema']),
      wire_hash=spec['request_hash'], mapping_version='evidence-integration-adapters-v1')

def _evidence_failure(reason,spec=None):
    r=_full_failure(reason,spec) if spec and spec['task']=='T2' else _object_failure(reason,spec)
    if spec:r.update(_evidence_metadata(spec))
    return r

def lossless(raw,aux_root):
    duplicates=[]; unavailable=[]; core=[]
    def walk(v,path):
        if isinstance(v,strict._ObjectPairs):
            groups={}
            for i,(k,x) in enumerate(v):groups.setdefault(k,[]).append((i,x))
            result={}
            for k,occ in groups.items():
                p=path+[k];resolved=[walk(x,p) for _,x in occ]
                if len(occ)>1:
                    aux=aux_root(p)
                    duplicates.append(dict(path=p,auxiliary=aux is not None,occurrences=[dict(ordinal=n+1,member_index=i,value=strict._lossless_value(x)) for n,(i,x) in enumerate(occ)]))
                    (core if aux is None else unavailable).append(p if aux is None else aux)
                result[k]=resolved[0] if len(occ)==1 else None
            return result
        if isinstance(v,list):return [walk(x,path+[i]) for i,x in enumerate(v)]
        return v
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',(raw or '').strip())
    obj=walk(json.loads(text,object_pairs_hook=strict._ObjectPairs),[])
    if core:raise ValueError('duplicate_core_paths:'+json.dumps(core))
    for path in unavailable:
        target=obj
        try:
            for key in path[:-1]:target=target[key]
            target[path[-1]]=None
        except (KeyError,TypeError):pass
    return obj,duplicates,unavailable

def _evidence_parse_t2(raw,spec):
    try:
        def auxiliary(p):
            if p and p[0]=='risk_state' and len(p)>1 and p[1] in ('reason','history_ids'):return ['risk_state',p[1]]
            if p and p[0]=='checks' and len(p)>2 and p[2] in ('reason','history_ids','final_ids'):return p[:3]
            return None
        obj,duplicates,unavailable=lossless(raw,auxiliary)
        if not isinstance(obj,dict) or set(obj)!={'risk_state','checks'}:raise ValueError('full_top_level_schema')
        state=obj['risk_state']
        if not isinstance(state,dict) or set(state)-{'value','reason','history_ids'} or not isinstance(state.get('value'),str) or state['value'] not in STATE_MAP:raise ValueError('risk_state_schema')
        mapped={'applicability':dict(state,value=STATE_MAP[state['value']]),'checks':obj['checks']}
        r=ng.parse('T2',json.dumps(mapped),spec,'NG1')
        r.update(_full_metadata(spec))
        r.update(technical_valid=r['valid'],target_evaluable=r['valid'],benchmark_scoreable=r['valid'] and r.get('applicability')=='applicable',raw_risk_state=state['value'],original_new_output=obj,ng1_mapping=mapped,duplicate_occurrences=duplicates,unavailable_auxiliary_paths=unavailable,verdict_status=r.get('verdict') if r['valid'] else 'technical_invalid')
        r['auxiliary_issues']=r.get('auxiliary_issues',[])+['duplicate_auxiliary:'+json.dumps(p) for p in unavailable];r.update(_evidence_metadata(spec));return r
    except (ValueError,TypeError,KeyError,AttributeError) as exc:return _evidence_failure(str(exc),spec)

OLD_CONTRACT='t2-risk-state-full-ng1-v1'
PRESENCE_CONTRACT='t2-f-condition-presence-ng1-v1'
MAPPING_VERSION='t2-forbidden-presence-fixed-map-v1'
F_MAP={'present':'yes','absent':'no','U':'U'}
def _t2_metadata(spec):
    return dict(explicit_contract=spec['contract'],representation_revision=spec['contract'],
                representation_hash=digest({'contract':spec['contract'],'schema':spec['output_schema'],'field_roles':spec.get('field_roles',{})}),scoring_definition_hash=digest(spec['definition']),
                schema_hash=digest(spec['output_schema']),wire_hash=spec['request_hash'],
                mapping_version=MAPPING_VERSION if spec['contract']==PRESENCE_CONTRACT else 'evidence-integration-adapters-v1')

def failure_t2(reason,spec=None):
    r=_evidence_failure(reason,spec)
    if spec:r.update(_t2_metadata(spec))
    return r

def auxiliary(path):
    if path and path[0]=='risk_state' and len(path)>1 and path[1] in ('reason','history_ids'):return path[:2]
    if path and path[0]=='checks' and len(path)>2 and path[2] in ('reason','history_ids','final_ids'):return path[:3]
    return None

def role_table(spec):
    roles=spec['field_roles']
    if not isinstance(roles,dict) or set(roles)!=set(spec['definition']['items']) or any(x not in ('required','forbidden') for x in roles.values()):
        raise ValueError('invalid_frozen_field_roles')
    return roles

def parse_t2(raw,spec):
    if spec.get('task')!='T2' or spec.get('contract') not in (OLD_CONTRACT,PRESENCE_CONTRACT):return failure_t2('unregistered_explicit_contract',spec)
    if spec['contract']==OLD_CONTRACT:
        r=_evidence_parse_t2(raw,spec);r.update(_t2_metadata(spec));return r
    try:
        roles=role_table(spec);obj,duplicates,unavailable=lossless(raw,auxiliary)
        if not isinstance(obj,dict) or set(obj)!={'risk_state','checks'}:raise ValueError('presence_top_level_schema')
        if not isinstance(obj['checks'],list):raise ValueError('presence_checks_not_list')
        mapped_checks=[];mappings=[];seen=set()
        for check in obj['checks']:
            if not isinstance(check,dict) or set(check)-{'id','value','final_ids','history_ids','reason'}:raise ValueError('presence_check_schema')
            id=check.get('id');value=check.get('value')
            if not isinstance(id,str) or id not in roles:raise ValueError('unknown_scoring_ID')
            if id in seen:raise ValueError('duplicate_scoring_ID')
            seen.add(id)
            allowed=F_MAP if roles[id]=='forbidden' else {'yes':'yes','no':'no','U':'U'}
            if not isinstance(value,str) or value not in allowed:raise ValueError('illegal_'+roles[id]+'_value:'+id)
            mapped=copy.deepcopy(check);mapped['value']=allowed[value];mapped_checks.append(mapped)
            mappings.append(dict(id=id,field_role=roles[id],wire_value=value,ng1_value=allowed[value]))
        # The inherited strict NG1 adapter validates state/coverage/NA/U/IDs and aggregates.
        # Its direct callable accepts this explicit contract and actual experimental arm.
        mapped=dict(risk_state=obj['risk_state'],checks=mapped_checks)
        r=_evidence_parse_t2(json.dumps(mapped,ensure_ascii=False),spec)
        r.update(presence_raw_output=raw,original_new_output=obj,presence_to_ng1_output=mapped,wire_new_values={x['id']:x['wire_value'] for x in mappings},field_value_mapping=mappings,
                 duplicate_occurrences=duplicates,unavailable_auxiliary_paths=unavailable)
        r['auxiliary_issues']=r.get('auxiliary_issues',[])+['duplicate_auxiliary:'+json.dumps(p) for p in unavailable]
        r.update(_t2_metadata(spec));return r
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        r=failure_t2(str(exc),spec);r['presence_raw_output']=raw;return r
