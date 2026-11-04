"""Frozen question schema and deterministic scoring rules for verifiable_t1."""
from __future__ import annotations

import json

import re

import time

from collections import Counter

from typing import Any, Dict, List, Optional, Tuple

_LIFT_STATUSES = {"left_attending", "cleared_by_doctor", "misdiagnosis", "removed"}

class QuestionContractError(ValueError):
    """Definition cannot supply the fields consumed by t1_check."""

def required_question_keys(gt_facts):
    return ({f"q_flag_{i}" for i in range(len(gt_facts.get("should_flag") or []))}
            | {f"q_lift_{i}" for i, diner in enumerate(gt_facts.get("diners") or [])
               if diner.get("status_change") in _LIFT_STATUSES})

def question_contract_errors(gt_facts, questions):
    questions = questions if isinstance(questions, dict) else {}
    return sorted(key for key in required_question_keys(gt_facts)
                  if not isinstance(questions.get(key), str) or not questions[key].strip())

def validate_question_contract(gt_facts, questions):
    if not isinstance(questions, dict):
        raise QuestionContractError("rubric_code_mismatch: questions_not_object")
    errors = question_contract_errors(gt_facts, questions)
    if errors:
        raise QuestionContractError("rubric_code_mismatch: " + ",".join(errors))

def t1_check(facts: Dict[str, Any], gt_facts: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Apply V5 rule: hard_trap + V_SAFE_FLAGGED + SOFT_R relaxation.

    SOFT_R relaxation: k_R = max(1, n_should_flag - 1) absorbs one noisy miss.
    safe_for_all is a veto (not in R count): only fail when agent ADDED warning.
    should_not_flag is hard-fail: not subject to k_R relaxation.
    """
    vetoes: List[str] = []
    hard_fail = False

    # Trap V3 (2026-05-24): per-diner full-lift detection (concrete diner +
    # extraction-disciplined yes/no), replacing the garbled abstract should_not_flag
    # triggers that flipped. Fires if the agent applied a fully-removed restriction
    # to any dish. Dropped: safe_for_all/no_match over-flag (option B), and ambiguous
    # 'corrected'/'takeaway_only' scope traps (inherently flippy).
    for i, d in enumerate(gt_facts.get("diners", []) or []):
        if (d.get("status_change") or "") in _LIFT_STATUSES and facts.get(f"q_lift_{i}") == "yes":
            vetoes.append(f"V_LIFT_{i}({d.get('name', '?')})")
            hard_fail = True

    n_should_flag = len(gt_facts.get("should_flag") or [])
    k_R_target = max(1, n_should_flag - 1) if n_should_flag else 0
    r_caught = 0
    r_miss: List[int] = []
    for i in range(n_should_flag):
        if facts.get(f"q_flag_{i}") == "yes":
            r_caught += 1
        else:
            r_miss.append(i)
    if n_should_flag and r_caught < k_R_target:
        vetoes.append(f"SOFT_R ({r_caught}/{k_R_target} of {n_should_flag} miss={r_miss})")

    passed = (not hard_fail) and (r_caught >= k_R_target if n_should_flag else True)
    return passed, vetoes
