#!/usr/bin/env python3
"""Report the string-typed argument fields in a BFCL dataset.

For each string field, prints its observed values, whether the schema already
marks it as an enum, and whether its values look like short labels (few words,
bounded length) rather than free text. Useful for deciding which fields could
be constrained with an `enum`.
"""

import argparse
import json
from collections import defaultdict

import bfcl_adapter as B


def looks_label(s):
    return len(s) <= 20 and len(s.split()) <= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    answers = {}
    for line in open(args.answer):
        line = line.strip()
        if line:
            a = json.loads(line)
            answers[a["id"]] = a

    field_values = defaultdict(list)
    field_has_enum = {}

    for line in open(args.question):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        if q["id"] not in answers:
            continue
        funcs = q["function"]
        if len(funcs) != 1:
            continue
        props = funcs[0]["parameters"].get("properties", {})
        try:
            _, args_dict = B.ground_truth_to_call(answers[q["id"]])
        except Exception:
            continue
        for field, spec in props.items():
            if isinstance(spec, dict) and spec.get("type", "").lower() in ("string", "str"):
                field_has_enum[field] = "enum" in spec
                v = args_dict.get(field)
                if v is not None:
                    field_values[field].append(str(v))

    print(f"Distinct string-typed fields: {len(field_values)}")
    n_enum = sum(1 for f in field_has_enum if field_has_enum[f])
    print(f"  already marked enum: {n_enum}\n")

    print(f"{'field':22s} {'n':>5} {'distinct':>8} {'enum?':>6} {'avg_len':>7}  sample")
    rows = sorted(field_values.items(), key=lambda kv: -len(kv[1]))
    for field, vals in rows[:args.show]:
        distinct = len(set(vals))
        avg_len = sum(len(s) for s in vals) / len(vals)
        max_len = max(len(s) for s in vals)
        marked = field_has_enum.get(field, False)
        frac_label = sum(looks_label(s) for s in vals) / len(vals)
        label_like = frac_label >= 0.8 and max_len <= 30
        sample = list(dict.fromkeys(vals))[:4]
        flag = " label-like" if (label_like and not marked) else (
            " (marked enum)" if marked else " free-text")
        print(f"{field:22s} {len(vals):5d} {distinct:8d} {str(marked):>6s} "
              f"{avg_len:7.1f}  {sample}{flag}")


if __name__ == "__main__":
    main()
