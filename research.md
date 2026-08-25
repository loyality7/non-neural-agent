# Non-Neural Paths to General Intelligence — Research Document

Status: Phase 0 done (literature). Experiment-0 built and run — **PASS** (see §12).
Sources archived in `research/sources/` (goal.txt, claude.md, chatgpt.md, qwen.md, nextplan.txt).

---

## 1. Question

> Can a persistent agent, using no gradient-based training on its reasoning stack,
> autonomously **discover**, **abstract**, and **transfer** a grounded object-affordance
> concept from raw sensorimotor experience?

Not "AGI without Transformers." Narrow, falsifiable, testable on a laptop.

## 2. Verdict from literature review

**Plausible, unproven, stuck at the same wall every time.**

- Substrate-independence of intelligence is theoretically sound (Church-Turing,
  functionalism, AIXI/Legg-Hutter). Not disproven.
- But every non-neural program that tried this — cognitive architectures (Soar,
  ACT-R, CLARION, LIDA, Sigma), artificial life (Tierra, Avida, PolyWorld),
  active inference, open-ended evolution (POET, MAP-Elites) — hit the **same
  bottlenecks**: representation discovery, combinatorial explosion, symbol
  grounding, catastrophic forgetting. Convergent failure = evidence the
  difficulty is fundamental, not incidental.
- Neural nets aren't proven *necessary* for reasoning/planning (Schema Networks,
  Bayesian Program Learning beat gradient nets at their own game in narrow
  domains) but are the best known cheap fix for **perception/grounding**.

## 3. Closest prior art (ranked, what to benchmark against)

**Update (2026-08-25) — a directly relevant paper published since the
original literature pass:** ONELIFE (Khan, Prasad, Stengel-Eskin, Cho,
Bansal; UNC/UT Austin/AI2; Oct 2025, [arXiv:2510.12088](https://arxiv.org/abs/2510.12088),
ICLR 2026 poster) is now the single closest published system to Phases 2-5
of this project, closer than QLAP. Confirmed real via direct search, not
taken on faith.

What it does: an agent explores a stochastic, hostile Crafter
reimplementation (Crafter-OO) with **no reward signal and no human-given
goals**, and reverse-engineers the world's causal rules from one unguided
life. Representation: modular precondition→effect "laws," each firing only
on the state attributes it governs (not the whole state at once — this is
their answer to the combinatorial-explosion problem this project hit in
Phase 4), multiple laws voting on a prediction when several apply, weighted
by learned confidence. World model then used for planning via rollout
simulation — comparing candidate multi-step strategies inside the learned
model. Their "Zombie Fighter"/"Stone Miner" scenarios are structurally the
same instrumental-value problem as this project's Phase 4/5 key-and-lock
case (a resource with zero direct reward, necessary for a downstream goal)
— they show their world model correctly ranks the multi-step plan above
the naive one. **This is independent, external validation that
value-propagation/forward-simulation over a discovered causal model is the
right mechanism for the exact gap Phase 4 surfaced and Phase 5 fixed** —
converging on the same design (backward/forward value propagation) from a
completely different implementation.

Where it genuinely differs, and where this project's novelty now sits
precisely, confirmed against the actual paper text (not just the abstract):
ONELIFE is LLM-driven at **two** separate stages, not one — (1) its
*exploration policy itself* is "driven by a large language model," given
"the high-level objective to discover as many underlying mechanics as
possible" rather than acting via any statistical curiosity signal, and (2)
its law *synthesizer* is a separate LLM call per observed transition,
proposing a candidate Python precondition/effect class from a state diff.
On top of both of those, its confidence-weight fitting is gradient-based
(L-BFGS, explicitly stated in their Sec. 3.4). So it is neurosymbolic in
three distinct places, not the strict non-neural, non-gradient system this
project is building — closer in spirit to PoE-World/WorldCoder than to
QLAP. **If this project's rule
discovery stays pure statistical counting/co-occurrence detection with
zero LLM or gradient involvement anywhere (true so far through Phase 5 —
`ConceptTable`/`ConceptGraph` are counts, sliding-window averages, and a
hand-coded Bellman backup, nothing learned via gradients) — that is now a
clean, citable differentiator**: "the closest published system to ours
(ONELIFE, 2025) uses an LLM for hypothesis generation and gradient-based
weight inference; this system performs both without any neural component."

Also newly relevant, more tangential: "Curious Causality-Seeking Agents in
Open-ended Worlds" (2025/2026) — directly relevant once Phase 6 (curiosity)
starts. Object-centric world models for causality-aware RL (AAAI 2026,
Nishimoto & Matsubara) and GRAIL (2026) push similar object-centric causal
structure but are neural-backed (attention/embeddings) — useful as
citations that the broader field agrees causal/object-centric structure
matters, not direct competition for the non-neural claim.

**Revised positioning:** the field has effectively split into (a) fully
neural/latent world models (Dreamer-style) and (b) neurosymbolic hybrids
leaning on an LLM for hypothesis generation once rule discovery gets hard
(ONELIFE, PoE-World, WorldCoder). Nobody currently publishing does rule/law
discovery through pure statistical/symbolic search with zero learned or
LLM components, combined with pre-registered thresholds/effect-size/seed-floor
rigor. That combination is the actual remaining white space — not because
it's unclaimed territory nobody thought of, but because groups solving this
problem reach for an LLM the moment hypothesis generation gets hard, which
is precisely the shortcut this project has been refusing to take (§8).

| Rank | System | What it did | Why it stalled |
|---|---|---|---|
| 1 | **QLAP** (Mugan & Kuipers 2012) | DBN + qualitative landmarks + intrinsic motivation; learns causal models + action hierarchy from continuous sensorimotor stream | Needed pre-built object trackers; hierarchy not modular |
| 2 | **Schema Mechanism** (Drescher 1991) | (context,action→result) schemas, invents "synthetic items" as abstractions | Combinatorial explosion beyond tiny grid-worlds |
| 3 | **IAC / Playground** (Oudeyer 2007) | Learning-progress curiosity, real robot, emergent affordance discovery | No abstraction/planning stack |
| 4 | **Schema Networks** (Kansky 2017) | Object-oriented causal PGM, zero-shot transfer, beat A3C | Needs entities handed to it |
| 5 | **DreamCoder** (Ellis 2020) | Wake-sleep program synthesis, invents reusable library functions | Not embodied, uses NN-guided search, pre-defined task domains |
| 6 | Bayesian Program Learning (Lake 2015) | One-shot concept learning beating deep nets | Domain-specific (Omniglot characters) |
| 7 | Soar / ACT-R | Full symbolic cognitive stack, chunking as learning | No autonomous concept emergence from raw experience — needs hand-coded ontology |
| 8 | Active inference (Friston) | Unifies perception/action/curiosity under one objective | Confined to small/discrete state spaces |

**The gap nobody has closed:** the *full* loop — raw perception → entity
segmentation → causal contingency discovery → abstraction into a reusable
named concept → zero/few-shot transfer — **without hand-given entity trackers**
and **with pre-registered, falsifiable emergence criteria**. QLAP and Schema
Networks both cheat on the entity layer. That's the opening.

## 4. Refined hypothesis (what we're actually testing)

> A non-gradient agent with (1) simple feature-based entity grouping,
> (2) explicit episodic memory, (3) statistical contingency detection,
> (4) a count-table "concept" store, and (5) short-horizon planning —
> can discover a hidden object→outcome rule from interaction alone, and
> apply that rule to a **novel instance** of the same object class it has
> never seen, at a rate significantly better than a memory-only agent and
> a random agent.

Neural components allowed **only** if raw-pixel perception is later
required and a non-neural clustering front-end fails there. Not needed for v1
(features are hand-readable, not raw pixels, on purpose — see §6).

## 5. What "emergence" means here (so we don't fool ourselves)

A result counts as genuine discovery only if:
- The agent was never given the label ("food", "red=good") — only raw features.
- The rule it forms is quantified over the **feature class**, not memorized
  per-instance (test: swap a red object to a new position/id → still works).
- It beats a memory-only baseline that has the exact same data but no
  abstraction mechanism.

If any of those fail, we don't get to call it emergence — that's the
discipline every prior report agreed on and it's the one rule we can't skip.

## 6. Minimal POC (Experiment-0) — what we're building now

**Environment** (`poc/env/gridworld.py`):
- 8×8 grid. Agent + a handful of colored objects (feature = color id, not pixels).
- Actions: `MOVE_N/S/E/W`, `EAT`.
- Hidden rule (agent never told): `color == RED and action == EAT → energy += 20`.
  All other colors/actions: energy -= 1 per step (upkeep cost), 0 net from eating.
- Episode ends at energy <= 0 or step cap; agent resets, memory persists.

**Agent** (`poc/agent/`):
- `memory.py` — ring buffer of `(observation, action, next_observation, Δenergy)`.
- `concepts.py` — count table keyed by `(feature, action) → outcome stats`;
  this *is* the abstraction step (grouped by feature, not by object identity).
- `planner.py` — depth-2/3 brute-force lookahead using the concept table.
- `agent.py` — wires perception → memory → concept update → planner → action.

**Baselines** (same episode budget, same env, different agent):
- `random` — no memory, acts randomly.
- `memory_only` — logs everything, never aggregates by feature (must match on
  exact remembered instance to act on it — no generalization possible).
- `full` — the proposed agent.

**Test** (`poc/experiments/exp0_entity_discovery.py`):
1. Run all three agents N episodes on the training layout.
2. Measure: steps until `P(energy+20 | red, eat)` estimate clears baseline
   with statistical margin (discovery speed).
3. Spawn a **new red object** at an unseen position/id never used in training.
   Check each agent's behavior toward it (transfer test).
4. Report survival time / energy-gain rate for all three, side by side.

## 7. Pass / fail — decided now, before running anything

**Pass** (hypothesis survives, worth extending):
- `full` agent's concept table converges on `red+eat→+energy` well above chance.
- `full` agent successfully eats the *novel* red object without having seen
  that exact instance before.
- `full` agent beats `memory_only` and `random` on survival time / energy rate.

**Fail** (documented as a real negative result, not hidden):
- Rule stays per-instance (no transfer to the new red object).
- `memory_only` matches `full` on the metrics — abstraction step bought nothing.
- No agent beats `random` — environment or reward signal is broken, fix before
  re-running, don't reinterpret the failure as success.

## 8. Explicitly out of scope for now

No neural nets, no raw pixels, no communication between agents, no
self-modification, no Crafter/MiniGrid, no multi-object causal conjunctions.
Those are Phase 4+ (see roadmap) — only after Experiment-0 passes.

## 9. Roadmap (only advance a phase if the previous one passed)

0. Literature + pre-registration — **done**
1. Entity/feature grouping + episodic memory + contingency table — **done, PASS (§11)**
2. Transfer test (novel instance, same feature) — **done, part of Experiment-0 (§11)**
3. Causal vs. correlation test (flip the rule mid-run, see if agent re-learns) — **done, PASS,
   large real effect (§12) — an earlier eyeballed "slow/thin" read was wrong and corrected in §12,
   don't cite that phrasing**
3.5. Adaptation-speed fix (sliding-window concept table) — **done, gate cleared cleanly (§14)**
4. Multi-object conjunctive rules (needs object A + object B) — **done (§15): representational
   fix confirmed (Cohen's d=0.616), and surfaced a distinct instrumental-value/subgoal gap
   handed to Phase 5**
5. Real planning — **not started. Redesign decision (plan.md §4): build value iteration over
   the concept graph as the primary mechanism (so value can propagate backward onto a
   zero-direct-reward enabler like the Phase-4 key), keep fixed-depth lookahead only as an
   explicit in-phase comparison baseline, not the fix itself. Gate: success rate on the
   Phase-4 key-then-red world rises well above the current 31% ceiling, 20 seeds, CI/effect size.**
6. Curiosity-driven exploration vs. random exploration — **not started. Per plan.md §4, test
   primarily on the Phase-4 conjunctive world (not a fresh simple world) — the key is rare and
   easy to miss by chance, exactly where directed exploration should show its biggest edge
   over random search.**
7. Raw pixel perception (likely break point — neural front-end may become necessary here)
8. Multi-agent / communication
9. Self-modeling / self-modification

## 10. Publication targets (once there's a real result, positive or negative)

IEEE ICDL, ALIFE conference, AGI conference (Springer LNAI), Frontiers in
Neurorobotics/AI. Not NeurIPS/ICML main track without a mature result and
strong baselines. arXiv (cs.AI/cs.NE) for preprint either way.

## 11. Experiment-0 result (run 2026-08-24, 20 fixed seeds)

Ran `poc/run.py`. Mean ± stdev across 20 seeds:

```
agent         survival(mean/std)    transfer_rate(mean/std)
random        65.09 / 0.95          0.10 / 0.06
memory_only   69.94 / 0.08          0.00 / 0.00
full          82.16 / 0.65          0.39 / 0.11

full's learned concept table: {red,EAT: +19.0, blue,EAT: -1.0, green,EAT: -1.0}
```

**PASS** on all three pre-registered criteria (§7), and stable — this replaces
the earlier single-run result which swung 0.07-0.47 on transfer rate alone
(that noise is now resolved by averaging over 20 seeds, per the falsification
criteria's own "≥20 seeds" bar):

- `full` correctly isolates `(red, EAT) → +19` with blue/green flat at -1 —
  confirms no color leakage, the concept table is doing real feature
  discrimination, not accidentally winning on one color.
- `full` transfers to a brand-new red object never seen during training at
  0.39±0.11 vs. `random` 0.10±0.06 and `memory_only` 0.00±0.00 — the
  `memory_only` zero is the structural point: it *cannot* generalize by
  design (exact-position memory only), so this comparison is the real
  evidence, not the raw survival numbers alone.
- `full` beats both baselines on survival time, consistently, low variance.

**Caveat, stated plainly:** smallest possible version of the claim. The
"abstraction" step groups by a hand-readable feature (color id), not raw
pixels — §6 says that's intentional for v1. Don't read this as a
raw-perception result; that's Phase 7, expected to be the hard part.

**Built-in vs. emergent (plan.md MVP criterion 8):** hand-given — the raw
color feature extraction, the concept table's structure (group by
(feature, action)), and the environment's reward rule itself. Never
given — the label "red = good" or any semantic name; the agent only ever
sees a raw color id and a resulting energy delta. Emergent: the actual
learned values (red→+19, blue/green→-1) and, from those, the transfer
behavior to a never-seen red object — none of that is programmed in.

Next real test per roadmap §9 is **Phase 3 — causal vs. correlation**: flip
the hidden rule mid-run and check whether `full` detects the change and
re-learns, rather than staying stuck on stale red-is-good belief.

## 12. Experiment-1 result (run 2026-08-24, 20 fixed seeds) — causal flip

`poc/experiments/exp1_causal_flip.py`: red is good for 200 episodes, then
green becomes good and red goes neutral for 200 more episodes, same agent,
memory never reset.

```
best rule correctly flips red -> green:            20/20 seeds
episodes needed to re-adapt (mean/std):             145 / 8.9
head-to-head (both visible) -- picks GREEN:         0.51 / 0.08
head-to-head (both visible) -- picks RED:           0.44 / 0.09
```

**PASS — slow to adapt, but the eventual preference is statistically real,
not noise.** Correction from an earlier draft of this section: a first pass
eyeballed the raw 51/44 head-to-head split and called it "thin, barely above
a coin flip." That was wrong, and it's exactly the mistake a proper
significance check exists to catch (plan.md, rigor standards). Re-run with
a bootstrap CI on the difference and Cohen's d (`poc/stats_utils.py`):

```
green - red head-to-head diff: 0.073, 95% bootstrap CI: [0.023, 0.123]
CI excludes zero: True
Cohen's d: 0.877 (large effect)
```

Per-seed variance is low (std ~0.08-0.09 across 20 seeds), so a 7-point raw
gap is a real, large, reliable effect, not sampling noise. Lesson kept on
the record on purpose: raw percentage gaps are not self-interpreting
without variance — always run the CI/effect-size check before characterizing
a margin as weak or strong.

What's genuinely still true: it takes ~145 of the 200 post-flip episodes to
flip its belief — that speed, not the eventual margin, is the real
weakness. Root cause, found by reading the concept table directly
(`green: count=392, mean=12.6` vs `red: count=703, mean=8.2` in one seed):
**`ConceptTable` is a lifetime cumulative average with no decay or forgetting.**
Every red-was-good sample from phase A stays in the average forever, so
red's value merely gets *diluted* by incoming neutral samples rather than
actually *replaced*. It works, but slowly and weakly — the exact
"catastrophic forgetting is the flip side of no-forgetting" trade-off the
literature review (§2, §3) predicted.

**Two test-harness bugs found and fixed while building this** (worth
recording because they'd have produced a false negative/positive):
1. First version of the transfer test built its throwaway test-world with
   the wrong (default) `good_color`, so the measured reward didn't match
   the actual post-flip rule at all — fixed by making transfer tests
   behavioral ("did it choose to eat this color") rather than
   reward-based, since the test world's own reward is not the thing under test.
2. First version of the transfer test placed only one candidate object, so
   "does it seek red" and "does it seek green" were independent questions,
   not a real preference comparison — an agent will happily seek *either*
   if its value estimate for that color is still above the eat-it threshold,
   even after the other color becomes strictly better. Fixed by adding a
   head-to-head test (both colors visible, equidistant) — that's the only
   version of this test that actually measures re-ranking, not just "still
   willing to bother."

**Built-in vs. emergent:** hand-given — the concept table structure and
the fact that the environment's rule changes at episode 200 (an
experimenter-controlled event, not something the agent does). Never
given — any signal that a flip occurred, or which color is now correct;
the agent has no "the rule just changed" input. Emergent — the entire
re-ranking (green overtaking red) is inferred purely from the agent's own
accumulating (diluted, in this version) observations of outcomes.

## 14. Phase 3.5 result — sliding-window concept table (run 2026-08-24)

Fix built: `ConceptTable` now keeps only the last 40 outcomes per
`(feature, action)` (a `deque(maxlen=40)`) instead of a lifetime running
average. Re-ran Experiment-1 unchanged otherwise, 20 seeds:

```
                          before (lifetime avg)   after (window=40)
episodes to re-adapt            145 / 8.9              25 / 8.9
head-to-head picks GREEN        0.51 / 0.08            0.64 / 0.08
head-to-head picks RED          0.44 / 0.09            0.16 / 0.08
Cohen's d                       0.877 (large)          6.075 (large)
```

Adaptation speed dropped from 145 to 25 post-flip episodes (~6x faster).
Stale red-preference actually decays now (0.44→0.16) instead of merely
being diluted — this is the qualitative fix the diagnosis in §12 called
for: the window lets old evidence age out completely rather than just
getting outweighed. Effect size went from already-large to enormous,
confirming the fix didn't trade margin for speed.

Re-ran Experiment-0 to confirm no regression: identical numbers (window
of 40 comfortably covers a single-rule 200-episode run, so this only
matters once the rule changes). **Phase 3.5 gate cleared cleanly.**

**Built-in vs. emergent:** hand-designed — the windowing mechanism itself
(fixed window size, sliding discard of the oldest sample). Emergent —
everything inside the window is still purely the agent's own observed
outcomes; the fix changes *how much history* is kept, not what's learned
from it.

Window size (40) was picked as a round number, not tuned — a real next
step if pursued further would be a small sweep (10/40/100) to check
sensitivity, but the gate is already cleared decisively enough that this
isn't blocking Phase 4.

**Sensitivity sweep, run later (2026-08-25), `poc/experiments/exp5_window_sweep.py`,
20 seeds per window size — checking this wasn't a fragile, cherry-picked
number before it anchors an MVP headline claim (plan.md §1.5 standard #6):**

```
window    reflip speed (mean/std)   green pref   red pref   Cohen's d
10        20 / 0.0                  0.62         0.15       8.201 (sig)
40        25 / 8.9                  0.58         0.17       5.813 (sig)
100       54 / 9.4                  0.65         0.15       5.154 (sig)
```

**Not fragile.** All three window sizes clear the significance gate
decisively (Cohen's d 5.1-8.2, all large, all significant) — window=40
wasn't a lucky pick, the fix works across an order of magnitude of the
hyperparameter. Sensible, interpretable pattern too: smaller window
re-adapts faster (10 → 20 episodes, essentially every seed converging at
the exact same speed, std=0.0) but that's expected — a shorter memory
forgets the stale rule almost as fast as it forgets anything; larger
window (100) re-adapts slower (54 episodes) because it needs more new
evidence to outweigh a longer memory of the old rule. window=40 sits in a
reasonable middle ground, not an outlier in either direction. This sweep
is now the citable answer to "why 40, not some other number."

## 15. Phase 4 result — conjunctive (AND) rule (run 2026-08-24, 20 seeds)

`poc/experiments/exp2_conjunctive_rule.py`, `poc/env/conjunctive_world.py`.
New hidden rule: eating RED only gives +20 if the agent already ate a
YELLOW key earlier in the same episode — a genuine two-condition rule the
Phase 1-3 `ConceptTable` (keyed by `(feature, action)` only, one color)
structurally cannot represent. Two agents compared:

- **flat** — `FullAgent`, completely unmodified, blind to whether it's
  holding the key.
- **conjunctive** — identical `ConceptTable`/planner code, but the feature
  string fed in is `f"{color}|key={has_key}"` instead of just `color` —
  the only change is a richer feature representation, not new machinery.

```
correct-behavior rate (eat red ONLY after grabbing key):
  flat (color only, blind to key):     0.26 / 0.09
  conjunctive (color + key context):   0.31 / 0.08
  random baseline:                     0.09 / 0.06

conjunctive beats flat: diff=0.053, 95% CI=[0.002, 0.107], Cohen's d=0.616 (medium)
conjunctive beats random: significant

flat's concept table (one seed):  {red,EAT: mean +2.0, others: -1.0}
conj's concept table (one seed):  {red|key=True,EAT: mean +19.0, red|key=False,EAT: -1.0, others: -1.0}
```

**Two separate findings here, not one — both real, neither smoothed over:**

1. **The predicted representational failure showed up exactly as expected.**
   `flat`'s concept table for `red,EAT` sits at a diluted, weakly-positive
   mean (+2.0 instead of +19) — it's mixing the lucky cases (had the key)
   with the unlucky ones (didn't) into one number, exactly the conflation
   the literature predicted (research.md §2-3, combinatorial/representational
   limits). `conjunctive`'s table is clean: `red|key=True` isolates to
   +19.0, everything else correctly flat. Giving the agent a richer
   feature representation — same code otherwise — measurably and
   significantly fixes this (medium effect size).

2. **But absolute success stays low (31%) even for the fixed agent, and
   that's a genuinely different problem, not the same one.** The key
   itself gives zero direct reward — it's a pure enabler. The greedy
   planner (`choose_action`) only walks toward objects with a learned
   positive value above threshold; since the key's own learned value stays
   near -1 (it never directly pays out), the agent has no mechanism to
   deliberately seek it out — it only picks it up by lucky random wandering.
   This is an **instrumental-value / subgoal-discovery gap**, distinct
   from the representational one: even with the correct concept
   (`red|key=True → good`), nothing propagates that value backward onto
   the enabling action (get the key). This is squarely a Phase 5 (real
   planning/lookahead) problem, not a Phase 4 one — noted here rather than
   patched, since patching it now would blur which fix solved which problem.

**Reading this honestly:** Phase 4's actual question — can a richer
feature representation capture a conjunctive rule where a flat one can't —
is answered **yes**, cleanly, with a real effect size. The low absolute
ceiling is a true and separate finding about what's still missing
(instrumental reasoning), not a failure of this phase's hypothesis.
Recorded as the concrete motivating case for Phase 5's "real planning"
gate, rather than left as an unexplained low number.

## 17. Phase 5 result — value iteration planning (run 2026-08-24, 20 seeds)

`poc/experiments/exp3_planning.py`, `poc/agent/value_iteration.py`. Per the
redesign decision in plan.md §4: built value iteration over a learned
2-state MDP (`has_key` in `{False, True}`) instead of fixed-depth lookahead.
Nothing about "the key enables red" is hand-given — the agent learns both
the reward table (`(feature, key_state) → mean delta_energy`, same sliding
window as Phase 3.5/4) and the key-flip transition probability
(`P(has_key becomes True | ate this feature while key=False)`) purely from
its own observed `has_key` before/after each EAT. Value iteration itself
(the Bellman backup, 50 sweeps, γ=0.9) is the one hand-designed mechanism —
tagged explicitly per plan.md MVP criterion 8.

Three agents on the identical Phase-4 world/test:
```
correct-behavior rate (eat red ONLY after grabbing key):
  flat (no key concept):                0.28 / 0.11
  conjunctive (correct rep, greedy):    0.31 / 0.06
  value_iter (correct rep, VI):         0.64 / 0.09

value_iter beats conjunctive-greedy: diff=0.333, 95% CI=[0.282, 0.382], Cohen's d=4.097
value_iter beats flat: significant
Q(yellow, key=False) = 169.1   Q(red, key=False) = 151.1   V(key=False)=169.0  V(key=True)=189.0
```

**PASS, cleanly, with the largest effect size seen in this project so far.**
`conjunctive` (Phase 4's fix) is reused here unmodified as the in-phase
comparison baseline plan.md called for, and it reproduces exactly the
ceiling Phase 4 found (~31%) — confirming that ceiling wasn't a fluke of
that experiment's setup. `value_iter` more than doubles it (64%). Critically,
`Q(yellow, key=False)` comes out positive and even *above* `Q(red,
key=False)` — the key gets correctly recognized as more valuable to pursue
than an unlocked red object, purely because value propagated backward from
the future reward it enables. That's the actual mechanism working, not
just an aggregate number moving.

**Honesty flag on the numbers, not the mechanism:** the raw Q/V magnitudes
(169, 189) are not physically meaningful — the 2-state MDP treats "eat
something" as available every abstract timestep with no accounting for the
real walking cost or episode length between opportunities, so value
iteration on an unbounded-horizon 2-state chain with γ=0.9 inflates without
bound in relative terms. What matters, and does hold up, is the *relative
ordering* (yellow ranks near/above red despite zero own reward) — that
ordering, not the absolute scale, is what the planner's threshold
comparison actually uses. Left in the record rather than rescaled away,
since a reviewer should be able to see this and confirm it's the ordering
doing the work, not an inflated number.

**Built-in vs. emergent, stated explicitly (plan.md MVP criterion 8):** the
2-state MDP structure and the value-iteration algorithm are hand-designed.
What's emergent: the actual reward values, the key-flip transition
probability, and therefore the entire resulting Q-ranking (including that
the key outranks red) — none of that is given, all of it is fit from the
agent's own experience.

## 18. Phase 6 result — curiosity-driven exploration (run 2026-08-25, 20 seeds)

`poc/experiments/exp4_curiosity.py`, `poc/agent/planner.py`
(`choose_action_curious`). Mechanism: count-based novelty — when nothing
currently visible clears the value threshold, walk toward whichever
visible object has been sampled fewest times instead of moving randomly.
Standard, simple, non-neural exploration signal.

**Confound found and fixed before the real test could run, kept in the
record (same practice as Experiments 1 and 4's earlier bugs):** the
Phase 4/5 conjunctive world places the key one step from spawn *by design*
("getting it is always physically easy — the question is whether the
agent learns it needs to," §15) — correct for isolating representation and
instrumental-value learning, but it means random wandering already finds
the key almost immediately regardless of strategy, leaving curiosity
nothing to improve on. First run on the unmodified Phase 4/5 world showed
curious and random performing identically at every checkpoint (as
expected once the confound is understood, not a real test of the
mechanism). Built `fresh_conjunctive_world_hard()` — key moved to the far
corner, genuinely outside the agent's `VISION_RADIUS=2` from spawn — and
re-ran on that instead.

**Result on the hard (far-key) world:**
```
episodes    curious            random             significant?
30          0.07 / 0.05        0.03 / 0.05        YES (d=0.805)
60          0.11 / 0.08        0.07 / 0.08        no  (d=0.522)
100         0.08 / 0.07        0.07 / 0.06        no  (d=0.155)
150         0.07 / 0.07        0.08 / 0.07        no  (d=-0.214)
200         0.06 / 0.04        0.08 / 0.07        no  (d=-0.271)
```

**FAIL on the pre-registered gate (curious must beat random at the final
checkpoint) — but the honest, more useful finding is in the shape of the
curve, not just the final row.** There IS a real, significant early edge
(30 episodes, d=0.805, large) — the mechanism does something. But it
washes out by 200 episodes, and both agents converge to a low ceiling
(~6-8%) regardless of exploration strategy.

**Root cause, diagnosed rather than left as an unexplained wash-out:**
`choose_action_curious` only re-ranks objects that are already *visible*
— it has no representation of "which parts of the grid I haven't searched
yet." With the key genuinely outside normal vision range for most of an
episode, curiosity has nothing to act on for the bulk of the time; both
agents are reduced to identical random movement until the key happens to
wander into view by chance, at which point the outcome is dominated by
random-walk statistics, not exploration strategy. The early transient
edge (30 episodes) is plausibly real signal on the occasions the key
*was* visible early — but it's a local, object-level novelty signal, not
a spatial/frontier exploration mechanism, and this world needs the latter.

**This is a genuine architectural gap, not a failure of the general idea:**
count-based novelty over *known, visible* things is a real, working
mechanism (Phase 4/5's key-adjacent-to-spawn setup, and the 30-episode
transient here, both show it can bias behavior correctly when there's
something to be curious about). What's missing is any notion of spatial
coverage or "go look in an unexplored region" — squarely out of scope for
this POC's design (§8: no raw pixels, no full spatial world model), but
worth stating precisely rather than reporting a vague "curiosity didn't
help." A real fix would need either a visited-cell count map (a legitimate,
still non-neural extension) or accepting that pure local novelty is
insufficient once the environment requires genuine spatial search — itself
a citable, specific negative result about the limits of this class of
exploration signal.

**Built-in vs. emergent:** the count-based preference rule itself is
hand-designed. What's emergent: the sample counts (and therefore which
specific visible object gets prioritized at any moment) — genuinely learned,
just insufficient here because the relevant unknown (where the key is
spatially) isn't represented by anything this agent tracks.

## 18b. Phase 6b — spatial (visited-cell) curiosity fix attempt (run 2026-08-25, 20 seeds)

Direct attempt to fix §18's diagnosed gap: added a visited-cell count map
(`poc/agent/planner.py`'s `choose_action_spatial_curious`) — when nothing
visible is worth acting on, walk toward whichever adjacent cell has been
visited fewest times, instead of the object-only novelty signal from Phase 6.

**One real bug found and fixed en route, kept in the record:** the first
version compared unclamped neighbor coordinates against the visit map — an
off-grid coordinate always reads as "unvisited" (count 0), so near any
edge the agent perpetually preferred stepping off the board, the
environment silently clamped it back, and it never escaped (signature:
exactly 0.0 success, zero variance, every single seed). Fixed by clamping
candidate cells to the grid the same way `GridWorld.step()` does.

**After the fix, still exactly 0.0 — but this time traced to a genuine
mechanism limit, not a bug, confirmed by direct trace:**
```
episodes  random         curious        spatial
30        0.03 / 0.05    0.07 / 0.05    0.00 / 0.00
200       0.08 / 0.07    0.06 / 0.04    0.00 / 0.00

spatial beats curious: False (Cohen's d=-2.078 -- spatial is WORSE)
spatial beats random:  False (Cohen's d=-1.665 -- spatial is WORSE)
```

Direct trace of a trained spatial agent (200 episodes) shows the fix
**does work for its intended purpose**: it finds the key at episode 0 —
faster than either `random` or `curious` ever manage on this world. The
learned Q-values are also correct: `Q(red, key=True)=189` clearly ranks
above everything else, exactly as Phase 5 designed. The failure is
downstream of both of those successes. Trace of an evaluation episode:
agent reaches the key in 8 steps, correctly eats it, and then oscillates
near the key's corner for the rest of the episode instead of heading
toward the far-corner red object, dying at step 79 having never eaten it.

**Root cause, diagnosed precisely:** two things compound. First, red is
not currently *visible* from where the agent sits after getting the key
(outside `VISION_RADIUS=2`), so the value-based branch of the planner
never fires — a correct Q-value is useless if the object it's about is
never in sight. Second, the spatial fallback that should send it looking
has degraded: visit counts are cumulative and **never reset** (same root
category of bug Phase 3.5 fixed for the concept table's reward memory —
recurring here in a different mechanism). After 200 episodes of thorough
exploration, the entire 8×8 grid has been visited roughly evenly, so
"go to the least-visited neighbor" no longer carries a coherent long-range
direction — it degenerates into local jitter once there's no more
meaningfully unvisited territory nearby to distinguish.

**The deeper gap this exposes, stated precisely:** the agent has no
persistent memory of *where* a valuable feature tends to be found — only
what that feature is worth. `ConceptGraph` tracks value per feature type,
never a feature's typical location. Fixing this properly would mean
adding spatial value memory (e.g., "red has historically been seen near
these coordinates") — a legitimate next mechanism, but a materially larger
addition than a visit-count table, and explicitly the kind of full
spatial/world-model machinery this POC scoped itself out of by design
(§8). Not built here; flagged precisely rather than attempted piecemeal.

**Standing back:** three real findings came out of chasing this one fix —
(1) count-based curiosity genuinely accelerates *initial* discovery (key
found at episode 0, confirmed by trace, not just inferred from the
aggregate numbers), (2) cumulative non-resetting counts saturate exactly
the way un-decayed reward memory did in Phase 3, a recurring pattern
across this project's mechanisms worth remembering, and (3) the actual
blocker for this specific world is spatial value memory, not exploration
strategy at all — a more precise diagnosis than "curiosity doesn't work,"
and a concrete, scoped target for anyone extending this later.

**Built-in vs. emergent:** hand-designed — the visited-cell counting
mechanism and the "prefer the least-visited neighbor" fallback rule
itself, plus the fixed `VISION_RADIUS=2` that determines what counts as
"currently visible." Emergent — the actual visit counts (which cells get
explored, when, relative to each other), and therefore which specific
direction gets prioritized moment to moment; the agent is never told
where the key or red objects are. What this experiment concretely shows
is *not* emergent, and would need to be: spatial value memory (§ above) —
correctly identified as a real missing built-in structure, not something
the current design could have produced by more training.

## 19. Tabular Q-learning baseline (run 2026-08-25, 20 seeds)

`poc/experiments/baseline_qlearning.py`. Standard epsilon-greedy tabular
Q-learning (state = position × current-cell-feature, α=0.1, γ=0.9,
ε=0.15, untuned — same standard held for our own hyperparameters), same
episode budget, same environment as Experiment-0. Purpose: internal
baselines (random, memory-only) prove our own components matter; they
don't prove this non-neural approach is competitive with the standard RL
alternative. Scope stated honestly: only Experiment-0's world, not
extended to Phase 4-6 environments (larger state space, would need a
separate convergence budget — a real, stated scope limit, not hidden).

```
agent           survival (mean/std)   transfer_rate (mean/std)
random          65.1 / 1.0            0.10 / 0.06
memory_only     69.9 / 0.1            0.00 / 0.00
full (ours)     82.2 / 0.6            0.34 / 0.09
q_learning      67.5 / 1.0            0.00 / 0.01
```

Our agent significantly beats Q-learning on both survival and transfer.
Exactly as expected, not a surprise dressed up as a finding: tabular
Q-learning has no mechanism to generalize across object identity — its
state includes exact grid position, so transfer to a novel red object at
an unvisited position requires having visited that exact state before,
structurally the same limitation as `memory_only`, for the same reason
(§11). This isn't "our method beats RL" in general — it's specifically
that feature-based abstraction (grouping by color, not position) is doing
real work that position-indexed Q-learning has no equivalent for, on this
exact task. Caveat kept in the record: Q-learning's larger state space may
need more than `TRAIN_EPISODES` to fully converge — reported at the same
budget as every other agent, not tuned in Q-learning's favor, so this
result should be read as "under matched sample budget," not as a claim
Q-learning cannot ever solve this task with enough data.

**Built-in vs. emergent:** hand-given — Q-learning's state representation
(position × feature) and hyperparameters (α, γ, ε), same standard applied
to our own untuned choices elsewhere. Emergent — the learned Q-table
values themselves. The key comparison point isn't built-in-vs-emergent at
all here (both agents learn purely from experience) — it's that our
concept table's *feature-only* keying is itself the hand-designed
structural choice that produces generalization Q-learning's
*position-inclusive* state representation structurally cannot, regardless
of how long either trains.

## 16. Standing rules (don't relitigate these)


- Pick one narrow claim, write the pass/fail number down first, then build.
- Always run baselines alongside the real agent — a number alone means nothing.
- Scale absurdly small until it works, then grow one dimension at a time.
- Report a failure exactly as found. A clean negative result is still a result.
- Neural nets aren't banned, they're just not the starting assumption — only
  reach for one when a non-neural piece demonstrably fails at something (§9 Phase 7).
