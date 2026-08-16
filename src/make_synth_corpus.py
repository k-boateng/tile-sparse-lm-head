#!/usr/bin/env python3
"""Generate a synthetic JSON corpus with a tunable free-text string fraction.

Emits JSON objects whose values are a mix of numbers, booleans, short tokens,
and free-text strings. `--string-fraction` controls how many values are
free-text sentences.
"""

import argparse
import json
import random
import string

KEYS = ["name", "id", "city", "status", "count", "email", "active", "score",
        "title", "category", "amount", "date", "type", "label", "value"]
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "the", "quick",
         "system", "request", "response", "user", "data", "field", "result"]


def rand_sentence(n):
    return " ".join(random.choice(WORDS) for _ in range(n))


def rand_value(string_fraction):
    if random.random() < string_fraction:
        return rand_sentence(random.randint(4, 20))
    r = random.random()
    if r < 0.4:
        return random.randint(0, 100000)
    if r < 0.6:
        return random.choice([True, False])
    if r < 0.75:
        return None
    return "".join(random.choice(string.ascii_lowercase)
                   for _ in range(random.randint(3, 8)))


def make_obj(string_fraction):
    keys = random.sample(KEYS, random.randint(2, 6))
    obj = {}
    for key in keys:
        v = rand_value(string_fraction)
        if random.random() < 0.15:
            v = {random.choice(KEYS): rand_value(string_fraction)}
        obj[key] = v
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--string-fraction", type=float, default=0.2)
    ap.add_argument("--out", default="data/synth.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    with open(args.out, "w") as f:
        for _ in range(args.n):
            f.write(json.dumps(make_obj(args.string_fraction)) + "\n")
    print(f"Wrote {args.n} objects (string_fraction={args.string_fraction}) -> {args.out}")


if __name__ == "__main__":
    main()
