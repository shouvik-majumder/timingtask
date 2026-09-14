# Exploration: why a probability floor and not an entropy bonus

## The failure it exists to prevent

The round-5 `act` run reached a 0.70 s delay with 100% of licks rewarded, held
it from update 525 to 875, and then froze — `corr` at 0.00 for 550 straight
updates with `eng` at 1.00, licking every trial and earning nothing, delay
pinned at 0.70.

It is not an incentive failure. At a 0.70 s delay the correct lick is worth
+13.25 and an early lick at 0.65 s is worth −3.62, a margin of **+16.9**. The
task overwhelmingly prefers waiting.

It is **saturation**. The score function that carries the whole policy gradient
is

    d log pi(lick) / d Delta = 1 - p

| p(lick) | 1 − p |
|---|---|
| 0.5 | 0.500 |
| 0.9 | 0.100 |
| 0.99 | 0.010 |
| 0.999 | 0.001 |

A policy that has become certain has almost no gradient however bad the outcome.
The −2.0 early penalty arrived on every trial and moved nothing.

## Why an entropy bonus cannot fix it

The entropy of a two-action policy is a function of the logit gap alone, and the
restoring force a bonus `−beta*Hbar` applies is

    d(Hbar)/d(Delta) = − Delta * p * (1 − p)

That is the **same vanishing factor**. The bonus peaks at `0.2239*beta` when
`|Delta| = 1.543` and decays like `beta*|Delta|*exp(−|Delta|)`; past `|Delta| ~ 4`
it is negligible. The term meant to prevent saturation dies at exactly the
moment saturation happens, for the same algebraic reason the gradient does.

It was never justified here anyway: imported from standard deep-RL practice,
with Song, Yang & Wang (2017) — whose setup this follows — setting theirs to
zero, and no paper on a withholding task reporting a sweep. **Removed.**

## What replaced it

A floor under every action probability, applied before sampling:

    p' = eps + (1 - 2*eps) * p          (eps = min_action_prob = 0.02)

The disfavoured action is therefore still sampled at least `eps` of the time,
and the score function for that sampled action is `~ p(1-p)/eps`:

| eps | score at p = 0.995 |
|---|---|
| 0 | 0.000 — the action is never sampled at all |
| 0.01 | 0.498 |
| 0.02 | 0.249 |
| 0.05 | 0.100 |

So a rare exploratory *wait* past the crossing carries both a usable gradient
and a large advantage. The mixed distribution is the one that acts and the one
whose log-probability is scored, so the policy gradient stays correct rather
than being taken with respect to a policy that was not followed.

Entropy is still logged. It is the clearest single readout of how deterministic
the policy has become — a measurement, not a term in the loss.

## The ablation

`act_nofloor` is `act` with `min_action_prob = 0`. If the floor is doing the
work, that run should reproduce the freeze: a delay that climbs and then stops
with `eng 1.00, corr 0.00` and entropy near zero.

Run it at more than one seed. The same configuration has produced cue-success
0.84 and complete failure, so a single run cannot separate an effect from the
seed.

## Round 6 as first run: void

The six round-6 variants (`act_g12`, `act_orth`, `act_noclip`, `act_step`,
`act_both`, `act_all`) were written as deltas from `BASE`, not from `act`.
`BASE` has `activity_penalty = 0.0`; `act` sets it to 20.0. So none of the six
carried the activity penalty — the only ingredient that had ever made a run
learn — and the sweep compared six `BASE` runs against one `act` run. Their
failures are the known `BASE` failure mode (never-lick collapse) and say
nothing about g, the recurrent initialisation, gradient clipping or the cue
shape. Discard those numbers.

Fixed by adding inheritance: a `RUNS` entry may name a `parent`, and every
round-6 entry now declares `parent="act"`. `run_all` prints a config matrix
before any training starts, listing every field that differs across the sweep,
so a column that varies when it was not meant to is visible before the compute
is spent rather than after.

## The censoring trap in the first-lick mean

A trial ends at `delay + answer_window`. The mean first-lick time therefore
rises with the delay even in an agent whose lick time does not move, because
the longer trial simply records late licks that the shorter one truncates.

Round 6, `act`: raw mean first lick 1.199 s at delay 1.0 against 1.507 s at
delay 1.8, which reads as timing. Restricted to licks earlier than 1.80 s --
the shorter trial's own deadline, a time both conditions had equal opportunity
to produce -- the two means are 1.198 and 1.187. Nothing moved. What actually
differed was the chance of licking at all: 0.38 against 0.56.

`summarise_by_delay` / `format_delay_report` (in `rl.py`) now compute the
capped mean and the slope `d(FL_cut)/d(delay)` -- 1.0 for an agent that times,
0.0 for one emitting a memorised interval -- and `run()` prints the table for
the training records, each test condition, the pooled fixed-1.0/fixed-1.8
pair, and the probe. Do not quote an uncapped first-lick mean across delays.
