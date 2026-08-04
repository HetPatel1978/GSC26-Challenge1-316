# Backdoor Attacks in Federated Learning — Remediation

IEEE GSC 2026, Challenge 1. Federated learning on CIFAR-10 with non-IID
clients, two backdoor attacks (BadNets, DBA), and two Byzantine-robust
aggregation defenses (Krum, FLTrust) evaluated against them.

## Problem statement

Federated learning lets clients train a shared model without sharing raw data,
but a malicious client can poison its local updates to implant a **backdoor**:
the global model behaves normally on clean inputs but misclassifies any input
carrying a fixed trigger pattern into an attacker-chosen target class. This
repo implements two attacks of increasing stealth, a training/eval pipeline
that measures them (accuracy on the real task vs. Attack Success Rate on
triggered inputs), and two Byzantine-robust aggregation defenses evaluated
against both.

## Approach

- **Framework**: [Flower](https://flower.ai/) (`flwr`) for the client/aggregation
  abstractions, PyTorch for the model.
- **Data**: CIFAR-10, partitioned across clients with a symmetric
  Dirichlet(α) distribution per class (`src/fl/data.py`) to control non-IID
  label skew — lower α means more skewed, non-IID client data.
- **Model**: a small 3-conv CNN (`src/fl/models.py`), sized so a full
  multi-round simulation with many clients finishes in minutes.

- **Attack — BadNets** (`src/attacks/badnets.py`): malicious clients patch a
  fixed 4x4 pixel trigger into a fraction of their local training images and
  flip the label to a fixed target class.
- **Attack — DBA (Distributed Backdoor Attack)** (`src/attacks/dba.py`): the
  same 4x4 trigger region is decomposed into four 2x2 quadrant pieces, and
  each malicious client only ever poisons its local data with its own single
  piece — no individual client's data (or gradient) ever contains the full
  pattern. Only the *aggregated* global model, having absorbed all four
  pieces across different clients/rounds, actually learns the complete
  trigger. This makes DBA materially harder to catch with per-client anomaly
  detection, since any one malicious update looks like a much smaller, subtler
  perturbation than a full BadNets update.
- **ASR measurement**: for both attacks, Attack Success Rate is measured
  identically — trigger every non-target-class test image with the *full*
  trigger pattern (the reassembled 4x4 pattern for DBA) and check what
  fraction the global model classifies as the target class. This keeps the
  two attacks' footprints directly comparable.

- **Defenses** (`src/defenses/aggregation.py`):
  - **FedAvg** — thin wrapper around Flower's own `aggregate` (plain
    weighted average, no robustness — the undefended control).
  - **Krum** — thin wrapper around Flower's own `aggregate_krum`: scores each
    client update by summed distance to its n-f-2 closest neighbors and keeps
    only the single most "central" update each round (single-Krum,
    `to_keep=0`), discarding the rest entirely.
  - **FLTrust** (Cao et al., 2021) — implemented directly (Flower doesn't
    ship one). The server holds a small trusted root dataset (500 samples,
    reserved via `reserve_root_set` in `src/fl/data.py` so it's strictly
    disjoint from every client's partition — never seen by any client). Each
    round the server trains a fresh copy of the global model on this root set
    to get a reference update direction `g0`. Every client update is
    ReLU-clipped cosine-scored against `g0` (a negative-similarity update
    gets zero trust, so it's fully excluded rather than merely down-weighted)
    and rescaled to `g0`'s norm — this specifically blocks the classic
    "boost your update's magnitude to dominate the average" scaling attack —
    then combined as a trust-weighted average of *all* clients, rather than
    Krum's keep-exactly-one-update approach.
- **Simulation driver** (`src/fl/experiment.py`): a sequential round loop that
  drives real `flwr.client.NumPyClient` instances and feeds their results
  into the aggregation functions above. **Note**: `flwr.simulation.start_simulation`
  (Ray-backed) is not used — Ray currently has no published wheel for
  Python 3.13, so the client-sampling / fit / aggregate / evaluate loop that
  Ray would otherwise orchestrate across actors is driven by hand instead.
  Everything else (the client abstraction, Flower's own aggregation
  implementations) is still genuine Flower code.

## Results

Setup (all 4 runs): 20 clients, Dirichlet α=0.5, 20% malicious (4 clients),
50% local poison rate, 30 rounds, 50% client participation per round.

| Attack | Defense | Final accuracy | Final ASR |
|---|---|---|---|
| BadNets | FedAvg (none) | 0.657 | 0.961 |
| BadNets | Krum | 0.316 | 0.000 |
| DBA | FedAvg (none) | 0.655 | 0.741 |
| DBA | FLTrust | 0.591 | 0.328 |

![Accuracy and ASR vs. rounds](results/plots/accuracy_asr_comparison.png)
![Final-round comparison](results/plots/final_round_comparison.png)

**FedAvg has no robustness mechanism at all**, so it can't tell a poisoned
update from a benign one: BadNets reaches 96% ASR (the full trigger is
present in every malicious client's gradient from round 1), while DBA climbs
more slowly and noisily to 74% (each client only ever contributes a quarter
of the pattern, so it takes longer for the aggregated model to piece the full
trigger together — but it still gets there).

**Krum vs. BadNets**: fully suppresses the attack (ASR → 0) by keeping only
the single most "representative" client update each round instead of
averaging. But under non-IID data that also means it's regularly discarding
good, unusual-but-honest updates along with the bad ones — accuracy plateaus
at 0.32, roughly half of FedAvg's. A known weakness of Krum: it trades away
most of federated learning's statistical efficiency to get robustness,
and that trade gets worse the more heterogeneous the clients are.

**FLTrust vs. DBA**: a much better accuracy/robustness trade — 0.591 accuracy
(only 6.6pp below the undefended run) with ASR cut from 0.741 to 0.328, a
>2x reduction. FLTrust doesn't drive ASR to zero the way Krum did against
BadNets: DBA's individual per-client updates are deliberately subtle (a
quarter of the trigger each), so they land closer to the trusted-direction
cosine similarity threshold than a full BadNets update would, and some still
pass the trust filter. This is exactly the intended difficulty comparison —
**DBA's whole design goal is to be harder for exactly this kind of per-update
anomaly filtering to catch**, and the results here reproduce that. It also
means Krum-vs-BadNets and FLTrust-vs-DBA aren't apples-to-apples on defense
strength alone; the interesting result is that no single defense tested here
gives both full suppression *and* full accuracy retention — a real
motivation for combining defense signals rather than relying on any one
mechanism.

Raw per-round metrics: `results/metrics/{fedavg_baseline,krum_defense,dba_fedavg,dba_fltrust}.json`.

## How to run

```bash
pip install -r requirements.txt

# BadNets attack
python scripts/run_fedavg_baseline.py   # no defense
python scripts/run_krum_defense.py      # Krum defense

# DBA attack
python scripts/run_dba_fedavg.py        # no defense
python scripts/run_dba_fltrust.py       # FLTrust defense

# Regenerate both comparison plots from results/metrics/*.json
python -m src.fl.plotting
```

GPU is used automatically if available (`torch.cuda.is_available()`); CIFAR-10
downloads to `./data/` on first run.

## Repo layout

```
src/fl/          data partitioning, model, Flower client, simulation driver, plotting
src/attacks/     BadNets and DBA trigger + poisoned-dataset logic
src/defenses/    aggregation strategies (FedAvg, Krum, FLTrust)
scripts/         one experiment per script (config + entry point)
results/metrics/ per-round JSON metrics for each run
results/plots/   generated comparison plots
```

## Status

Attack x defense matrix implemented and evaluated: BadNets x {FedAvg, Krum},
DBA x {FedAvg, FLTrust}. FLAME defense was scoped as a stretch goal and was
cut to keep the four combos above fully evaluated and documented ahead of the
deadline.
