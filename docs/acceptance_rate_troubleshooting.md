## EAGLE3 Acceptance Length (AL) Troubleshooting Guide

Target: push acceptance length (avg accepted tokens per verify round) toward 2.0–3.0 on your data. Your baseline: AL≈1.5 (epoch 3–7 on Llama 405B) vs ~2.5 on a Llama‑4 run.

### Quick checklist (run these first)
- [x] Spec params at inference: `--speculative-num-steps <= ttt_length`, avoid exceeding training value
- [x] Try non‑default spec triples (examples below)
- [x] Zero temperature, no penalties at verify; greedy verification
- [ ] Template/headers/EOT match exactly; masks non‑trivial and only assistant spans
- [x] Same tokenizer/model across conversion → extraction → training → inference
- [x] Vocab mapping coverage of masked tokens ≥ 95%
- [x] Effective batch sufficiently large (via accumulation) and LR not too high [BS 16 was used]
- [x] Enough data (≥30k samples if possible) and domain mix similar to inference [20K is enough]

---

## Common root causes and fixes

### 1) Speculation settings mismatch
- Symptom: Low AL despite good draft accuracy.
- Causes:
  - `speculative-num-steps` > trained `ttt_length` (e.g., steps=6 with ttt=5)
  - Too many draft tokens with small top‑k → verifier rejects later tokens
  - Non‑greedy verify (temperature/penalties active)
- Fixes:
  - Keep `speculative-num-steps <= ttt_length` (match training value)
  - Tune `(steps, topk, draft_tokens)`; start from `(5,4,8)` then try:
    - Higher acceptance: `(4,3,6)` # Already tried this.
    - Conservative/low latency: `(3,2,4)`
    - Throughput‑leaning: `(6,4,10)` (watch AL and OOM)
  - Verify with `temperature=0`, no repetition/top‑p during verification

### 2) Formatting / masking issues
- Symptom: Draft trains on wrong spans; AL capped.
- Causes: header‑only assistant turns, missing EOT between turns, labels counted as content, template mismatch.
- Fixes:
  - Embed `Agent:`/`Patient:` in headers (custom template) so labels are not masked
  - Ensure exactly one `<|eot_id|>` between turns; final EOT optional
  - Drop/merge header‑only assistant segments
  - Validate with loss‑mask visualization and sum stats

### 3) Tokenizer/template inconsistency across stages
- Symptom: Good train acc, poor AL at inference.
- Causes: different tokenizer or chat template in any stage.
- Fixes: pin the same tokenizer path and template name for conversion, extraction, training, inference.

### 4) Vocab mapping coverage (t2d/d2t)
- Symptom: Verifier rejects due to draft vocabulary mismatch.
- Causes: mapping built on different data; low coverage of masked tokens; wrong target/draft sizes.
- Fixes:
  - Rebuild mapping from the exact train set used; verify coverage: masked token coverage ≥ 99.5%
  - Keep `draft_vocab_size` consistent with config

### 5) Data volume/domain
- Symptom: AL ~1.5 plateaus.
- Causes: 20k samples insufficient, domain shift.
- Fixes: push to ≥30k samples; diversify prompts; balance lengths.

### 6) Optimization setup
- Symptom: Noisy updates, slow convergence.
- Causes: small effective batch, LR too high for width, short warmup, weak clipping.
- Fixes:
  - Increase `draft_global_batch_size` via accumulation (keep micro=1 to fit VRAM)
  - LR scaling: for 405B (hidden_size=16384), start 2e‑5; probe 3e‑5 with warmup 0.05 and clip 0.3
  - Keep `ttt_length` constant (raising increases memory)

### 7) Aux hidden layers / target head
- Symptom: High variance, poor alignment to target logits.
- Causes: wrong aux layer indices, target head mismatch.
- Fixes: confirm `eagle_aux_hidden_state_layer_ids` and target head loading keys; verify shapes.

### 8) Length effects
- Symptom: AL lower on long sequences.
- Causes: draft degrades later in sequence; truncation at 2048.
- Fixes: report AL by position/length; optionally cap speculation beyond N tokens or reduce draft_tokens for long tails.

### 9) Dtype/TP/runtime
- Symptom: Intermittent acceptance drops.
- Causes: kernel differences (sdpa vs flex), TP gather/all‑reduce inconsistencies.
- Fixes: stick to one backend (flex), ensure TP all‑reduce on outputs is correct (done in code), avoid mixed tokenizers.

---

## Diagnostics

### A) AL sweep over spec params
```bash
# Pseudocode: run short eval and log AL for a grid
for steps in [3,4,5,6]:
  for topk in [2,3,4,5]:
    for draft in [4,6,8,10,12]:
      launch with --speculative-num-steps $steps --speculative-eagle-topk $topk --speculative-num-draft-tokens $draft
      measure mean AL on 200 prompts; record (steps,topk,draft,AL,latency)
```

### B) Vocab mapping coverage
```python
import torch
def coverage(dataset, t2d):
    # fraction of masked tokens that are in draft vocab
    num, den = 0, 0
    for item in dataset:
        ids = item["input_ids"].view(-1)
        mask = item["loss_mask"].view(-1).bool()
        den += int(mask.sum())
        num += int(torch.isin(ids[mask], t2d.nonzero().view(-1)).sum())
    return num/ max(1,den)
```

### C) Mask sanity
```python
# Expect only assistant spans green; sum > 0 and not tiny
```

---

## Recommended non‑default spec combos (not (5,4,8))
- Higher acceptance: `--speculative-num-steps 4 --speculative-eagle-topk 3 --speculative-num-draft-tokens 6`
- Conservative latency: `--speculative-num-steps 3 --speculative-eagle-topk 2 --speculative-num-draft-tokens 4`
- Throughput‑leaning: `--speculative-num-steps 6 --speculative-eagle-topk 4 --speculative-num-draft-tokens 10`

---

## Action plan (given your notes)
1) Align inference to trained `ttt_length=5`; run the non‑default spec sweeps above; pick the best AL/latency pareto.
2) Increase effective batch (global=16, micro=1) — already done; if still noisy, try global=24–32 with same micro.
3) LR: keep 2e‑5 to 3e‑5 with warmup≥0.05 and clip≤0.3; only test 5e‑5 after stability verified.
4) Expand data toward 30k+ (your L4 reference used 30k).
5) Verify vocab mapping coverage ≥ 99.5% on masked tokens.
6) Confirm template/mask invariants are satisfied (already validated).


