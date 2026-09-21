# AIShield: Adaptive Red Team / Blue Team Benchmark for Prompt Injection Defense

AIShield is a benchmark and research harness for studying adversarial prompt-injection attacks and adaptive defense. It models the core loop of a security contest:

- a red team generates attacks that try to jailbreak or override an AI system,
- a blue team firewall evaluates those prompts and decides whether to allow, warn, or block them,
- successful evasions are fed back into the blue team’s learned corpus,
- the next round of attacks evolves to exploit the weaknesses the system has not yet learned to block.

The project is designed to be both an experimental benchmark and a technical reference implementation for adversarial robustness research in LLM safety settings.

---

## Why this project matters

Modern LLM systems are vulnerable to prompt injection, jailbreaks, role-play hijacks, indirect instructions, multi-turn escalation, and context poisoning. Many naive defenses focus on obvious phrases like “ignore previous instructions,” but that fails under mutation, paraphrase, obfuscation, and conversation-based escalation.

This repository is built around a simple but important idea:

> A defense is only credible if it is tested against an evolving attacker that learns from what already got through.

That is the novelty of the project: it turns prompt injection defense into a closed-loop adversarial benchmark instead of a static classifier demo.

The system does not just evaluate a firewall once. It repeatedly asks:

1. How do attack prompts evolve?
2. What patterns survive the defender?
3. What new examples should the firewall learn?
4. Does the defense improve under adaptive pressure?

This makes it useful for research, benchmarking, and experimentation with defense strategies.

---

## Project novelty

The key novelty is the combination of all of the following:

### 1. Closed-loop red team vs blue team simulation
The project explicitly models a co-evolving adversarial loop:

- attacker generates risky prompts,
- defender scores them,
- successful attacks are added to the defensive corpus,
- the next generation of attacks adapts to previously successful evasion.

This is more realistic than a one-off static evaluation because it captures the adversarial arms race between attacker and defender.

### 2. Mutation-based adversarial evolution without requiring a paid model
The red team is not limited to static templates. It can mutate existing prompts using transformations such as:

- homoglyph substitution,
- leetspeak,
- zero-width character insertion,
- base64-wrapped payloads,
- spacing-based evasion,
- unnatural insertion of benign text around an attack payload.

This creates a low-cost yet effective adversary even when no API key or large model is available.

### 3. Multi-layer defense, not just keyword blocking
The firewall is not a single regex list. It combines:

- lexical detection,
- semantic similarity detection,
- context-aware conversation analysis.

This better reflects how real prompt injection attacks work: they often look harmless in isolation but become malicious across turns or in combination with surrounding context.

### 4. Adaptive learning via confirmed evasion memory
The blue team stores confirmed evasions in a corpus and uses them to score future prompts. This is important because it creates a self-improving defense without needing to retrain a model from scratch.

### 5. Transparent benchmark design
The project evaluates both:

- attack success rate (ASR),
- false-positive rate (FPR),
- detection rate,
- corpus growth,
- cross-round trends.

This matters because a model that blocks everything but also blocks benign prompts is not a useful defense.

---

## Core idea in one sentence

The repository is a benchmark environment for testing whether a defense can adapt to attacks that mutate, obfuscate, and escalate across dialogue turns while still preserving acceptable utility on benign prompts.

---

## Architecture

The project is split into two major subsystems:

```text
redteam/     -> attack generation and adversarial evolution
blueteam/    -> detection, scoring, and learning from confirmed attacks
```

### Red team
The attacker side is implemented in `redteam/` and includes:

- `seed_library.py`: handcrafted jailbreak templates and prompt seeds,
- `dataset_library.py`: public prompt datasets and offline fallback samples,
- `mutator.py`: transformations to obfuscate and mutate adversarial prompts,
- `llm_client.py`: Anthropic-based generation of new attack variants,
- `local_llm_client.py`: local OpenAI-compatible model support,
- `generator.py`: orchestrates seeding, mutation, and evolutionary progression,
- `schema.py`: data structures for attacks and evaluation results.

### Blue team
The defender side is implemented in `blueteam/` and includes:

- `normalize.py`: de-obfuscation and normalization before scoring,
- `lexical_layer.py`: phrase and keyword detection,
- `semantic_layer.py`: similarity scoring vs known attacks and benign examples,
- `context_layer.py`: assessment of conversation-level escalation or poisoning,
- `corpus.py`: persistent learned corpus of confirmed attacks and benign negatives,
- `firewall.py`: the main policy engine that combines all layers into a final decision,
- `schema.py`: structured result objects and explanatory metadata.

---

## What the system actually does

Each benchmark round follows this loop:

1. Generate a batch of attack prompts.
2. Normalize and analyze them with the blue team firewall.
3. Measure which attacks successfully evade detection.
4. Add successful evasions to the learned corpus.
5. Evolve the next batch using mutation and model-guided generation.
6. Repeat for several rounds.

This is implemented in `full_loop.py`, which is the main execution entry point.

---

## The main loop

The main execution is in `full_loop.py`.

It does the following:

- initialize the red team generator,
- initialize the blue team firewall,
- create the initial seed attack batch,
- evaluate each attack,
- flag any attack that was allowed,
- confirm those evaded prompts into the firewall corpus,
- evolve the next generation,
- repeat for a fixed number of rounds.

The script also computes metrics including:

- Attack Success Rate (ASR)
- Detection Rate
- False Positive Rate (FPR)
- Multi-turn FPR
- Corpus size

It writes results to JSON files such as `full_loop_metrics.json` and `full_loop_corpus.json`.

---

## Red team generation strategy

The attacker is intentionally designed to be cheap and flexible.

### Seed generation
The initial round uses attack templates from `seed_library.py`, plus optional public dataset examples from `dataset_library.py`.

These seeds cover categories such as:

- direct override
- roleplay/persona hijacking
- obfuscation and encoding
- indirect injection
- multi-turn escalation
- context poisoning
- tool abuse

### Mutation generation
Once the first round has run, the red team mutates surviving attacks using programmatic obfuscation. This is a core part of the benchmark because it tests whether a defender generalizes beyond exact phrase matching.

### LLM-guided generation
If configured, the system can call an Anthropic API or a local OpenAI-compatible server to generate novel variants inspired by previously successful attacks.

This allows the benchmark to go beyond “static mutation” and simulate a more sophisticated adversary.

---

## Blue team detection strategy

The firewall is designed to be a layered detector rather than a one-shot rule matcher.

### 1. Normalization layer
Before scoring, the text is normalized to remove or reverse common obfuscation. This includes:

- Unicode normalization,
- zero-width stripping,
- homoglyph mapping,
- leetspeak conversion,
- spacing collapse,
- base64 decoding where appropriate.

This is essential because many attack prompts are not direct natural-language requests; they are disguised or encoded.

### 2. Lexical layer
The lexical layer is fast and transparent. It looks for known suspicious phrases and structures.

This catches obvious jailbreak patterns like direct instruction override attempts.

### 3. Semantic layer
The semantic layer compares candidate text against known attack and benign examples. It uses one of multiple backends:

- TF-IDF ensemble
- sentence-transformer embeddings (if configured)
- Anthropic judge backend (if configured)

This helps detect paraphrased or disguised prompts that do not literally contain a known attack phrase.

### 4. Context layer
A major differentiator is the context layer. It does not evaluate only the current message; it assesses the full conversation trajectory.

It looks for:

- multi-turn escalation,
- context poisoning,
- abrupt topic shift,
- conversational drift into malicious intent.

This is important because a prompt that is harmless by itself can become dangerous when preceded by earlier malicious turns.

### 5. Final decision
The firewall combines scores into a single risk estimate and maps it to a decision:

- allow,
- warn,
- block.

The result object is structured and explainable, which is useful for research and debugging.

---

## Learnable corpus and adaptation

One of the most important parts of the project is the corpus in `blueteam/corpus.py`.

The corpus contains examples labeled as:

- attack,
- benign.

It starts with built-in examples and then grows as confirmed attacks are added. This is the adaptation mechanism that makes the defense smarter over time.

This is conceptually similar to a memory bank of previously detected evasion attempts.

### Why this matters
Without a learned corpus, a firewall may catch obvious patterns but fail on mutated variants. With a corpus, it gains the ability to recognize near repeats and similar adversarial patterns.

---

## Experiments included in the repo

The benchmark includes multiple reproducible experiments:

### 1. Full closed-loop run
```bash
python full_loop.py --rounds 6
```
Runs the complete adaptive benchmark.

### 2. Statistical multi-run benchmark
```bash
python run_n_times.py --n-runs 10 --rounds 6 --base-seed 42
```
Measures performance across repeated trials and produces summary statistics.

### 3. Matched two-arm comparison
```bash
python compare_arms.py --n-runs 10 --rounds 6 --base-seed 42 --ablation lexical_only --driver adaptive
```
Compares a static lexical baseline to the adaptive system on the same attack batches.

### 4. Mutation-depth experiment
```bash
python run_depth_experiment.py --n-runs 10 --rounds 6 --base-seed 42
```
Studies how mutation depth affects attacker success and whether capping depth helps the defense.

### 5. Publication figure generation
```bash
python generate_publication_figures.py
```
Produces publication-quality plots for attack success rate, false positives, category mitigation, and corpus growth.

### 6. Verification of generated benchmark results
```bash
python verify_report.py --results-doc RESULTS.md
```
Checks the reported metrics against raw logs.

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the default closed-loop benchmark
```bash
python full_loop.py --rounds 6
```

### 3. Run with optional LLM-based variant generation
```bash
set ANTHROPIC_API_KEY=your_key_here
python full_loop.py --rounds 6 --use-llm
```

### 4. Skip the public dataset and use only seed templates
```bash
python full_loop.py --rounds 6 --no-dataset
```

### 5. Compare real firewall vs dummy fallback
```bash
python run_round.py
python run_round.py --real-firewall
```

---

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── RUN_COMMANDS.txt
├── requirements.txt
├── full_loop.py
├── run_round.py
├── run_n_times.py
├── compare_arms.py
├── run_depth_experiment.py
├── full_loop_metrics.json
├── benchmark_results.json
├── compare_arms_metrics.json
├── mutation_depth_comparison.json
├── held_out_benign.py
├── real_blue_team.py
├── dummy_blue_team.py
├── redteam/
│   ├── __init__.py
│   ├── corpus_store.py
│   ├── dataset_library.py
│   ├── generator.py
│   ├── llm_client.py
│   ├── local_llm_client.py
│   ├── mutator.py
│   ├── schema.py
│   ├── seed_library.py
│   └── ...
├── blueteam/
│   ├── __init__.py
│   ├── context_layer.py
│   ├── corpus.py
│   ├── firewall.py
│   ├── known_patterns.py
│   ├── lexical_layer.py
│   ├── normalize.py
│   ├── schema.py
│   ├── semantic_backends.py
│   ├── semantic_layer.py
│   └── ...
├── publication_figures/
│   └── table_benchmark_summary.tex
├── tests/
│   └── test_design_changes.py
└── ...
```

---

## Technical strengths

This project is valuable because it captures several practical lessons:

- attack generation can be automated cheaply,
- prompt injection is often not a single obvious phrase,
- defensive systems need contextual understanding,
- adversaries evolve, so defense must also evolve,
- benchmark design matters as much as model quality.

It is therefore less of a “toy demo” and more of a benchmark framework for adversarial LLM safety research.

---

## Known limitations

This project is intentionally transparent about its constraints. The main ones are:

- the default TF-IDF semantic backend is still lexical-heavy and may miss true paraphrase-only evasions,
- the real Hugging Face dataset path is not fully validated in constrained environments,
- mutation can sometimes produce a temporary ASR rebound when the red team re-mutates batches after survivor starvation,
- benchmark results depend on the quality and coverage of the held-out benign set.

These limitations are discussed in the project documentation and are important context for interpreting the benchmark results.

---

## Best use cases

This project is best suited for:

- security research on prompt injection,
- benchmarking LLM defense strategies,
- testing adversarial robustness under mutation and obfuscation,
- exploring adaptive security as a closed-loop process,
- building a reproducible research pipeline for AI safety evaluation.

---

## License and safety note

This project is intended for controlled evaluation of defensive systems in research and benchmarking settings.

It is not intended for unauthorized exploitation of third-party services or for generating harmful content at scale. The repository is structured as a security research harness and should be used responsibly.

---

## Summary

AIShield is a practical and research-oriented benchmark for prompt-injection defense in LLM systems. Its main contribution is not simply a firewall implementation, but a full adaptive adversarial loop:

- generate attacks,
- detect them,
- learn from misses,
- evolve the attacker,
- evaluate the defender again.

That feedback loop is what gives the project its research value and makes it more than a static prompt filter.

If you want to understand the system, start with:

1. `full_loop.py`
2. `redteam/generator.py`
3. `blueteam/firewall.py`
4. `blueteam/corpus.py`
5. `RESULTS.md`

These files contain the actual logic of the red team, blue team, and overall benchmark loop.

