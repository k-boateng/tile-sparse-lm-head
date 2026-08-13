import argparse
import json
import sys
from pathlib import Path
 
import xgrammar as xgr
from transformers import AutoTokenizer


def build_matcher_factory(tokenizer_name : str):
    """
        Loads tokenizer, build XGrammar tokenizer info + compiled JSON Grammar

        Returns (make_matcher, tokenizer, vocab_size). make_matcher() yields a
        fresh GrammarMatcher so each document starts from the initial state.
    """

    tok = Autotokenizer.from_pretrained(tokenizer_name)
    #full vocab size (not tok.vocab size) matches model output dims

    ti = xgr.TokenizerInfo.from_huggingface(tok)
    compiler = xgr.GrammarCompile(ti)

    grammar = compiler.compile_builtin_json_grammar() #generic JSON
    #can be swapped later

    vocab_size = ti.vocab_size

    def make_matcher():
        return xgr.GrammarMatcher(grammar)
    
    return make_matcher, tok, vocab_size

def walk_document(doc_string, tokenizer, make_matcher, vocab_size, bitmask):
    """Walk one output string through the grammar, yielding per-step valid sets.
 
    Yields (step_index, valid_ids_list) for each decode step. Harvests the
    mask BEFORE accepting each token. The mask describes what's legal at the
    *current* state, i.e. the choice the model faced at that step.
    """
    token_ids = tokenizer.encode(doc_string, add_special_tokens=False)
    matcher = make_matcher()

    for step, tid in enumerate(token_ids):

        #what is legal right now?
        matcher.fill_next_token_bitmask(bitmask)
        bool_mask = xgr.testing.bitmask_to_bool_mask(bitmask, vocab_size)
        valid_ids = bool_mask.nonzero().flatten().tolist()
        yield step, valid_ids

        #accept the actual next token.
        #if not valid, stops and accpet fails
        if not matcher.accept_token(tid):
            print(
                f"  [warn] token {tid} ({tokenizer.decode([tid])!r}) rejected at "
                f"step {step}; grammar/corpus mismatch, truncating doc.",
                file=sys.stderr,
            )
            return

def load_corpus(path):
    """Load corpus strings. Supports:
    - .jsonl where each line is a JSON object -> we re-serialize the object as
      the output string to walk.
    - .txt where each line is already the raw output string.
    Adjust to your BFCL extraction as needed.
    """
    path = Path(path)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                obj = json.loads(line)
                # Re-serialize compactly; this is the structured OUTPUT string.
                out.append(json.dumps(obj, separators=(",", ":")))
            else:
                out.append(line)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B",
                    help="HF tokenizer id. Ungated fallback: Qwen/Qwen2.5-7B")
    ap.add_argument("--corpus", required=True,
                    help="Path to corpus (.jsonl of objects, or .txt of strings)")
    ap.add_argument("--out", default="valid_sets.jsonl",
                    help="Output .jsonl of per-step valid token sets")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Cap number of documents (for a quick first pass)")
    args = ap.parse_args()
 
    print(f"Loading tokenizer and grammar: {args.tokenizer}")
    make_matcher, tok, vocab_size = build_matcher_factory(args.tokenizer)
    print(f"vocab_size = {vocab_size}  (tiles of 256 -> "
          f"{-(-vocab_size // 256)} total tiles)")
 
    corpus = load_corpus(args.corpus)
    if args.max_docs:
        corpus = corpus[: args.max_docs]
    print(f"Loaded {len(corpus)} documents.")
 
    bitmask = xgr.allocate_token_bitmask(1, vocab_size)
 
    n_steps = 0
    with open(args.out, "w") as fout:
        for doc_i, doc in enumerate(corpus):
            for step, valid_ids in walk_document(
                doc, tok, make_matcher, vocab_size, bitmask
            ):
                fout.write(json.dumps({
                    "doc": doc_i,
                    "step": step,
                    "n_valid": len(valid_ids),
                    "valid_ids": valid_ids,
                }) + "\n")
                n_steps += 1
            if (doc_i + 1) % 50 == 0:
                print(f"  {doc_i + 1}/{len(corpus)} docs, {n_steps} steps")
 
    print(f"Done. {n_steps} decode steps harvested to {args.out}")
 
 
if __name__ == "__main__":
    main()