#!/usr/bin/env python3
"""Build BFCL (schema, output) pairs with enums injected on string fields.

For each string-typed field whose observed values form a small set of short
labels, an `enum` of those values is added to the schema before compilation.
Fields with many or long values are left as free strings.
"""

import copy
import json
from collections import defaultdict

import bfcl_adapter as B

MAX_ENUM_DISTINCT = 15
MAX_ENUM_LEN = 20
MAX_ENUM_WORDS = 3


def collect_field_value_sets(question_path, answer_path):
    """Gather observed string values per (func_name, field)."""
    answers = {}
    for line in open(answer_path):
        line = line.strip()
        if line:
            a = json.loads(line)
            answers[a["id"]] = a

    fieldvals = defaultdict(set)
    for line in open(question_path):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        if q["id"] not in answers:
            continue
        funcs = q["function"]
        if len(funcs) != 1:
            continue
        fname = funcs[0]["name"]
        props = funcs[0]["parameters"].get("properties", {})
        try:
            _, args = B.ground_truth_to_call(answers[q["id"]])
        except Exception:
            continue
        for field, spec in props.items():
            if isinstance(spec, dict) and spec.get("type", "").lower() in ("string", "str"):
                v = args.get(field)
                if isinstance(v, str) and v != "":
                    fieldvals[(fname, field)].add(v)
    return fieldvals


def should_enum(values):
    if not values or len(values) > MAX_ENUM_DISTINCT:
        return False
    return all(len(v) <= MAX_ENUM_LEN and len(v.split()) <= MAX_ENUM_WORDS
               for v in values)


def enrich_function_schema(func, fieldvals):
    """JSON schema string with enums injected on enumerable string fields."""
    fname = func["name"]
    params = copy.deepcopy(func["parameters"])
    for field, spec in params.get("properties", {}).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("type", "").lower() in ("string", "str"):
            vals = fieldvals.get((fname, field), set())
            if should_enum(vals) and "enum" not in spec:
                spec["enum"] = sorted(vals)
    schema = B._normalize_schema(params)
    if schema.get("type") != "object":
        schema["type"] = "object"
    return json.dumps(schema)


def load_enriched_bfcl_pairs(question_path, answer_path, max_entries=None):
    """Yield (schema_json_str, output_json_str) pairs with enum enrichment."""
    fieldvals = collect_field_value_sets(question_path, answer_path)
    n_enum = sum(1 for v in fieldvals.values() if should_enum(v))
    print(f"[enrich] {n_enum}/{len(fieldvals)} string fields marked as enums")

    answers = {}
    for line in open(answer_path):
        line = line.strip()
        if line:
            a = json.loads(line)
            answers[a["id"]] = a

    pairs = []
    skipped = 0
    for line in open(question_path):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        if q["id"] not in answers:
            continue
        funcs = q["function"]
        if len(funcs) != 1:
            continue
        try:
            schema = enrich_function_schema(funcs[0], fieldvals)
            _, args = B.ground_truth_to_call(answers[q["id"]])
            output = json.dumps(args, separators=(",", ":"))
        except Exception:
            skipped += 1
            continue
        pairs.append((schema, output))
        if max_entries and len(pairs) >= max_entries:
            break
    print(f"[enrich] built {len(pairs)} pairs, skipped {skipped}")
    return pairs
