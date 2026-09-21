# AIShield — Red Team Attack Generator

Implements the "AI Red Team" and "Attack Technique Library" components
described in the AIShield project report (Sections 8 and 13). This is
the attacker half of the Red Team vs. Blue Team benchmark — it doesn't
depend on the Blue Team firewall existing yet, so you can build and test
it in isolation before wiring the two together.

## What's here

```
redteam/
  schema.py            Attack / AttackResult / GenerationBatch data models
  seed_library.py       Hand-written seed templates, 7 attack categories
  dataset_library.py     Loader for a public jailbreak-prompt dataset
                          (Hugging Face), with an offline fallback sample
  mutator.py             Cheap, no-LLM-needed mutations (homoglyphs, base64,
                          zero-width spaces, leetspeak, sandwiching, spacing)
  llm_client.py           Anthropic API wrapper for generating fresh variants
  local_llm_client.py      Optional client for a local OpenAI-compatible
                            model server (Ollama, vLLM, LM Studio, etc.)
  generator.py             RedTeamGenerator — orchestrates seeding, LLM
                            generation, mutation, and round-over-round evolution
  corpus_store.py          JSON-lines storage for the growing attack corpus
dummy_blue_team.py       Trivial keyword-based stand-in Blue Team, so the
                          pipeline runs before your real firewall exists
run_round.py              Runnable end-to-end demo of the round-based
                           benchmarking loop (seed -> mutate -> evolve)
generate_three_batches.py  Generates three separate JSON files in one pass:
                            dataset-sourced, mutator-derived, and LLM-generated
requirements.txt
```

## Generating three separate attack batches

```bash
python generate_three_batches.py
```

Produces three files:

| File | Source | Needs a model? |
|---|---|---|
| `dataset_attacks.json` | Public jailbreak dataset (Hugging Face `rubend18/ChatGPT-Jailbreak-Prompts` by default) | No |
| `mutator_attacks.json` | Programmatic mutations applied to the dataset attacks | No |
| `llm_attacks.json` | Freshly LLM-generated variants | Yes — Anthropic API or a local model |

**Dataset step:** tries `pip install datasets` + a real Hugging Face pull
first. If that package isn't installed, or there's no network, it
automatically falls back to a small offline sample of paraphrased,
publicly-documented jailbreak families (DAN, AIM, STAN, etc.) so the
rest of the pipeline still runs. Swap `DATASET_ID` in
`redteam/dataset_library.py` for a different HF dataset if you'd rather
use another corpus.

**LLM step:** checks for a model in this order and writes an **empty
list** if none is found —

1. `ANTHROPIC_API_KEY` env var set → uses the Anthropic API
2. `LOCAL_MODEL_ENDPOINT` env var set (e.g. `http://localhost:11434` for
   Ollama, or wherever your OpenAI-compatible local server listens) →
   uses that local model instead. Optionally set `LOCAL_MODEL_NAME` too.
3. Neither set → `llm_attacks.json` is written as `[]`, with a note
   printed to the console.

```bash
# with Hugging Face + Anthropic:
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python generate_three_batches.py --n-dataset 50 --n-per-category 5

# with a local model instead (e.g. Ollama running llama3.1):
export LOCAL_MODEL_ENDPOINT=http://localhost:11434
export LOCAL_MODEL_NAME=llama3.1
python generate_three_batches.py
```

## Quick start

```bash
pip install -r requirements.txt   # only needed for --use-llm mode or real dataset pulls
python run_round.py --rounds 3
```

Round 1 now combines two sources by default: hand-written seed
templates (`seed_library.py`) and real prompts pulled from a public
jailbreak dataset (`dataset_library.py` — falls back to an offline
sample automatically if `datasets` isn't installed or there's no
network). Each round after that evolves whatever evaded the previous
round via mutation, regardless of which source it originally came
from — the console output shows the source breakdown per round, e.g.:

```
Round 1: 23 attacks [dataset=8, seed=15], Attack Success Rate = 69.6%
Round 2: 32 attacks [mutated=32], Attack Success Rate = 100.0%
```

Flags:
- `--no-dataset` — Round 1 uses hand-written seeds only (fully offline, fastest)
- `--n-dataset N` — cap how many dataset prompts get pulled into Round 1
- `--use-llm` — also generate fresh LLM variants each round (Anthropic API if `ANTHROPIC_API_KEY` is set, else a local model if `LOCAL_MODEL_ENDPOINT` is set, else this flag is a no-op)

To also generate fresh LLM-authored variants each round:

```bash
export ANTHROPIC_API_KEY=sk-...
python run_round.py --rounds 5 --use-llm
```

## Plugging in the real Blue Team

`dummy_blue_team.py` is the only file that should change once your real
firewall exists. Replace its `evaluate()` function with a call to your
actual firewall (HTTP request to your FastAPI service, or a direct
Python import if it's in the same codebase). It must return a list of
`AttackResult` — see `redteam/schema.py` for the exact fields
(`decision`, `risk_score`, `evaded`).

```python
from redteam.schema import AttackResult

def evaluate(attacks):
    results = []
    for attack in attacks:
        response = requests.post("http://localhost:8000/firewall/check",
                                  json={"turns": attack.turns}).json()
        results.append(AttackResult(
            attack_id=attack.id,
            decision=response["decision"],       # "allow" | "warn" | "block"
            risk_score=response["risk_score"],
            evaded=(response["decision"] == "allow"),
        ))
    return results
```

## Programmatic usage

```python
from redteam import RedTeamGenerator, CorpusStore

gen = RedTeamGenerator()                       # no llm= → seed/mutation only
store = CorpusStore("my_corpus.jsonl")

batch = gen.generate_seed_batch(round_number=1)
store.save_batch(batch)

# ... evaluate batch.attacks against your Blue Team, get `results` back ...

next_batch = gen.evolve_round(batch.attacks, results, round_number=2)
```

## Extending the attack library

Add new categories or templates in `redteam/seed_library.py`. Multi-turn
attacks are just a list of strings instead of one string — see
`MULTI_TURN_ESCALATION` and `CONTEXT_POISONING` for examples; the Blue
Team should replay each string as a successive user turn in the same
conversation.

## Notes

- Payloads in `seed_library.DEFAULT_PAYLOADS` are deliberately abstract
  (system-prompt disclosure, policy bypass) — the point is to test the
  firewall's *structural* detection, not to produce genuinely harmful
  content. Keep it that way if you extend the library.
- This tool is meant to run against a system you own and are
  benchmarking (your own AIShield firewall). It isn't intended for use
  against third-party systems you don't have authorization to test.
