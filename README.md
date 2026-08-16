# tile-sparse-lm-head

Measuring vocabulary-tile sparsity in the LM-head projection during
grammar-constrained decoding.

## Idea

During grammar-constrained decoding (JSON, tool calls), the grammar marks most
of the vocabulary invalid at each step, yet the LM head still projects the
hidden state over the full vocabulary. Because invalid tokens receive logit −∞
(so `e^(−∞) = 0`), omitting their rows of the unembedding matrix is exact.

If the vocabulary is permuted offline into contiguous 256-token tiles grouped by
syntactic category, a decode step only needs the tiles that contain at least one
grammar-valid token. This repo measures how many tiles are active per step
across real workloads — the quantity that determines how much of the projection
can be skipped.

## What the scripts measure

`measure.py` walks structured outputs through an [XGrammar](https://github.com/mlc-ai/xgrammar)
matcher and, at each decode step, records:

- `n_valid` — number of grammar-valid tokens
- `identity` — active tiles under the original vocabulary order
- `category` — active tiles under a category-grouped permutation
- `lb` — lower bound, `ceil(n_valid / 256)`
- `state` — a coarse grammar-state label (structural / number / string / …)

It reports the distribution of active-tile fractions and a per-state breakdown.

## Setup
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt      # xgrammar, transformers, numpy, matplotlib
pip install bfcl-eval                 # for the BFCL data
```

The measurement is CPU-only; no GPU is required.

## Running

Generic JSON grammar over a synthetic corpus:

```bash
python src/make_synth_corpus.py --n 200 --string-fraction 0.2 --out data/synth.jsonl
python src/measure.py --tokenizer Qwen/Qwen2.5-7B --corpus data/synth.jsonl \
    --out results/measure_synth.jsonl
```

Per-entry schema-derived grammars from BFCL:

```bash
BFCL=$(python -c "import bfcl_eval, os; print(os.path.dirname(bfcl_eval.__file__))")/data
python src/measure.py --tokenizer Qwen/Qwen2.5-7B \
    --bfcl $BFCL/BFCL_v4_simple_python.json $BFCL/possible_answer/BFCL_v4_simple_python.json \
    --out results/measure_bfcl.jsonl
```

Same, with enums injected on enumerable string fields:

```bash
python src/measure.py --tokenizer Qwen/Qwen2.5-7B \
    --bfcl-enriched $BFCL/BFCL_v4_simple_python.json $BFCL/possible_answer/BFCL_v4_simple_python.json \
    --out results/measure_bfcl_enriched.jsonl
```

Inspect the string-typed fields in a BFCL dataset:

```bash
python src/diagnose_strings.py \
    --question $BFCL/BFCL_v4_simple_python.json \
    --answer $BFCL/possible_answer/BFCL_v4_simple_python.json
```

## Files

```
src/measure.py           the measurement (all modes)
src/bfcl_adapter.py      BFCL function definitions -> JSON schema + reconstructed call
src/enrich_schema.py     inject enums on enumerable BFCL string fields
src/make_synth_corpus.py synthetic JSON corpus generator
src/diagnose_strings.py  report string-typed fields in a BFCL dataset
```

`measure.py` writes one JSON record per decode step to `--out`; the summary
statistics are printed to stdout (pipe through `tee` to keep them).
