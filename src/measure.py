#!/usr/bin/env python3
"""Convert BFCL `simple` entries into (json_schema, output_string) pairs.

Each BFCL entry pairs a function definition with a ground-truth call. This
builds a JSON schema from the function's parameters and reconstructs the
arguments object as a compact JSON string, taking the first acceptable value
for each argument. Entries that fail to convert are skipped and counted.
"""

import json

_TYPE_MAP = {
    "dict": "object", "float": "number", "integer": "integer", "int": "integer",
    "string": "string", "str": "string", "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array", "tuple": "array", "number": "number",
    "any": "string",
}


def _normalize_schema(node):
    """Rewrite BFCL type names into JSON Schema types, recursively."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                mapped = _TYPE_MAP.get(v.lower())
                if mapped is None:
                    raise ValueError(f"unmapped type: {v}")
                out[k] = mapped
            elif k in ("properties", "items", "additionalProperties"):
                out[k] = _normalize_schema(v)
            else:
                out[k] = _normalize_schema(v) if isinstance(v, (dict, list)) else v
        return out
    if isinstance(node, list):
        return [_normalize_schema(x) for x in node]
    return node


def function_to_schema(func):
    """BFCL function dict -> JSON schema string."""
    schema = _normalize_schema(func["parameters"])
    if schema.get("type") != "object":
        schema["type"] = "object"
    return json.dumps(schema)


def first_value(acceptable):
    """Take the first non-empty acceptable value for an argument."""
    if not isinstance(acceptable, list):
        return acceptable
    for v in acceptable:
        if v != "":
            return v
    return acceptable[0] if acceptable else None


def ground_truth_to_call(gt_entry):
    """{"ground_truth":[{func:{arg:[vals]}}]} -> (func_name, {arg: value})."""
    gt = gt_entry["ground_truth"][0]
    func_name = next(iter(gt))
    args = {arg: first_value(vals) for arg, vals in gt[func_name].items()}
    return func_name, args


def load_bfcl_pairs(question_path, answer_path, max_entries=None):
    """Yield (schema_json_str, output_json_str) pairs from BFCL files."""
    answers = {}
    with open(answer_path) as f:
        for line in f:
            line = line.strip()
            if line:
                a = json.loads(line)
                answers[a["id"]] = a

    pairs = []
    skipped = {"no_answer": 0, "bad_schema": 0, "reconstruct": 0}
    with open(question_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            qid = q["id"]
            if qid not in answers:
                skipped["no_answer"] += 1
                continue
            funcs = q["function"]
            if len(funcs) != 1:
                skipped["bad_schema"] += 1
                continue
            try:
                schema = function_to_schema(funcs[0])
            except Exception:
                skipped["bad_schema"] += 1
                continue
            try:
                _, args = ground_truth_to_call(answers[qid])
                output = json.dumps(args, separators=(",", ":"))
            except Exception:
                skipped["reconstruct"] += 1
                continue
            pairs.append((schema, output))
            if max_entries and len(pairs) >= max_entries:
                break

    print(f"[bfcl] loaded {len(pairs)} pairs, skipped {skipped}")
    return pairs
