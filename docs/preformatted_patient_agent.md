## Preformatting Patient–Agent Dataset for Offline EAGLE3 (Llama 3)

- [x] Goal: Convert turn-structured conversation JSONL into a single-text JSONL that matches the custom template for correct loss masking and hidden-state extraction.

### Inputs and Assumptions

- [x] Source JSONL (turn-structured): `/fsx/brayden/SpecForge/cache/regen-dataset/I52_L2_sll_23k_kickout_1k_magpie_50k_h20_52k_shuf.converted_regen_i60.jsonl`
- [x] Target model/tokenizer: Llama 3/405B tokenizer at `/fsx/coreweave-training/release/i60`
- [x] Custom template name: `llama3-preformatted-patient-agent`
- [x] Template strings (must match exactly):
  - [x] User header: `<|start_header_id|>user<|end_header_id|>\n\nPatient: `
  - [x] Assistant header: `<|start_header_id|>assistant<|end_header_id|>\n\nAgent: `
  - [x] End-of-turn: `<|eot_id|>`
- [x] System prompts are variable; no fixed system prompt in the template.

### Why a single preformatted text per row

- [x] Loss masking finds assistant spans by searching the exact header and EOT tokens in one contiguous string, then maps character spans back to tokens using offsets. Splitting headers across multiple fields makes span detection brittle; a single `text` guarantees stable, correct masks.

---

## Step 1 — Build preformatted JSONL (single `text` per row)

- [x] Create a conversion that turns each row’s `conversations` into one `text` string.
- [x] If `content` already includes Llama headers/EOT, concatenate in order and ensure an `<|eot_id|>` after every turn.
- [x] Otherwise, wrap each user/assistant message with the registered headers and append `<|eot_id|>`.

```python
from datasets import load_dataset
import json, os

SRC = "/fsx/brayden/SpecForge/cache/regen-dataset/I52_L2_sll_23k_kickout_1k_magpie_50k_h20_52k_shuf.converted_regen_i60.jsonl"
DST = "/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl"

USER_HDR = "<|start_header_id|>user<|end_header_id|>\n\nPatient: "
ASSIST_HDR = "<|start_header_id|>assistant<|end_header_id|>\n\nAgent: "
EOT = "<|eot_id|>"

def format_turn(role, text):
    if role == "user":
        return f"{USER_HDR}{text}{EOT}"
    if role == "assistant":
        return f"{ASSIST_HDR}{text}{EOT}"
    return ""  # ignore other roles or extend if needed

ds = load_dataset("json", data_files=SRC)["train"]

def to_text(batch):
    out = []
    for conv in batch["conversations"]:
        parts = []
        for m in conv:
            role = m.get("role", "")
            content = m.get("content", "")
            # If content already has headers/EOT, keep as-is and ensure EOT
            if "<|start_header_id|>" in content:
                seg = content
                if not seg.rstrip().endswith(EOT):
                    seg += EOT
                parts.append(seg)
            else:
                parts.append(format_turn(role, content))
        out.append("".join(parts))
    return {"text": out}

ds = ds.map(to_text, batched=True)
ds = ds.remove_columns([c for c in ds.column_names if c != "text"]) 
os.makedirs(os.path.dirname(DST), exist_ok=True)
ds.to_json(DST, lines=True)
print("Wrote:", DST)
```

- [x] Output JSONL path: `/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl`

### Concrete commands (for your dataset)

- [x] Convert your current conversations file to single-text JSONL (appends `<|eot_id|>` when missing, including assistant turns that lack a terminal EOT):

```bash
python3 - <<'PY'
from datasets import load_dataset
import os

SRC = "/fsx/brayden/SpecForge/cache/regen-dataset/I52_L2_sll_23k_kickout_1k_magpie_50k_h20_52k_shuf.converted_regen_i60.jsonl"
DST = "/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl"
EOT = "<|eot_id|>"

def ensure_eos(s): 
    return s if s.rstrip().endswith(EOT) else s + EOT

ds = load_dataset("json", data_files=SRC)["train"]

def to_text(batch):
    out = []
    for conv in batch["conversations"]:
        joined = "".join(seg.get("content", "") for seg in conv)
        out.append(ensure_eos(joined))
    return {"text": out}

ds = ds.map(to_text, batched=True)
ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
os.makedirs(os.path.dirname(DST), exist_ok=True)
ds.to_json(DST, lines=True)
print("Wrote:", DST)
PY

```

- [x] Extract hidden states on the preformatted file:

```bash
sudo mkdir -p /opt/dlami/nvme/i60-hidden-states-masked
sudo chown "$USER":"$USER" /opt/dlami/nvme/i60-hidden-states-masked
setsid nohup torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py -- \
  --dist-timeout 12000 \
  --model-path /fsx/coreweave-training/release/i60 \
  --enable-aux-hidden-states \
  --disable-custom-all-reduce \
  --data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --is-preformatted \
  --chat-template llama3-preformatted-patient-agent \
  --max-length 2048 \
  --tp-size 8 \
  --batch-size 1 \
  --mem-frac 0.725 \
  --num-samples 20000 \
  --output-path /opt/dlami/nvme/i60-hidden-states-masked/ \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  >> /opt/dlami/nvme/i60-hidden-states-masked/llama_405b.out 2>&1 < /dev/null &
```

- [ ] Train offline on the saved states:

```bash
torchrun --nproc_per_node=8 scripts/train_eagle3_offline.py \
  --target-model-path /fsx/coreweave-training/release/i60 \
  --draft-model-config configs/llama3-405B-eagle3.json \
  --train-data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --train-hidden-states-path /opt/dlami/nvme/i60-hidden-states \
  --chat-template llama3-preformatted-patient-agent \
  --is-preformatted \
  --output-dir /fsx/brayden/outputs/i60-lr-2e5 \
  --num-epochs 10 --draft-global-batch-size 4 --draft-micro-batch-size 1 \
  --learning-rate 2e-5 --max-length 2048 --ttt-length 5 --tp-size 8
```

---

## Step 2 — Sanity validations (on preformatted JSONL)

### 2.1 Header and EOT coverage

- [ ] Every row contains the user header, assistant header, and at least one EOT.

```python
from datasets import load_dataset

DST = "/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl"
USER_HDR = "<|start_header_id|>user<|end_header_id|>\n\nPatient: "
ASSIST_HDR = "<|start_header_id|>assistant<|end_header_id|>\n\nAgent: "
EOT = "<|eot_id|>"

ds = load_dataset("json", data_files=DST)["train"]

def check(batch):
    ok_user = ok_assist = ok_eot = 0
    n = len(batch["text"])
    for t in batch["text"]:
        ok_user += int(USER_HDR in t)
        ok_assist += int(ASSIST_HDR in t)
        ok_eot += int(EOT in t)
    return {"ok_user": [ok_user], "ok_assist": [ok_assist], "ok_eot": [ok_eot], "n": [n]}

agg = ds.map(check, batched=True, batch_size=1000, remove_columns=ds.column_names)
tot = {"ok_user": 0, "ok_assist": 0, "ok_eot": 0, "n": 0}
for r in agg:
    for k in tot: tot[k] += r[k]
print(tot)
assert tot["ok_user"] == tot["n"] and tot["ok_assist"] == tot["n"] and tot["ok_eot"] == tot["n"], "Header/EOT coverage failed"
```

### 2.2 Assistant payload presence

- [ ] Assistant spans (between assistant header and next user header/EOT) contain non-empty payload.

```python
import re
from datasets import load_dataset

DST = "/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl"
USER_HDR = "<|start_header_id|>user<|end_header_id|>\n\nPatient: "
ASSIST_HDR = "<|start_header_id|>assistant<|end_header_id|>\n\nAgent: "
EOT = "<|eot_id|>"
USER_SEP = EOT + USER_HDR
ASSIST_SEP = EOT + ASSIST_HDR

ds = load_dataset("json", data_files=DST)["train"]

def payload_ok(batch):
    bad = 0
    for text in batch["text"]:
        # spans following ASSIST_SEP
        pattern = re.escape(ASSIST_SEP) + r"(.+?)(?=" + re.escape(USER_SEP) + r"|$)"
        spans = list(re.finditer(pattern, text, flags=re.DOTALL))
        # handle first assistant if text starts with ASSIST_HDR
        if text.startswith(ASSIST_HDR):
            first_span = text[len(ASSIST_HDR):]
            first_span = first_span.split(USER_SEP)[0]
            spans = spans + ([type("obj", (), {"group": lambda _ : first_span})],)
            # hacky append: only for sanity, skip in production
        for m in spans:
            if len(m.group(0 if hasattr(m, 'group') else 0).strip()) == 0:
                bad += 1
    return {"bad": [bad], "rows": [len(batch["text"])]}

agg = ds.map(payload_ok, batched=True, batch_size=500, remove_columns=ds.column_names)
bad = sum(r["bad"] for r in agg)
rows = sum(r["rows"] for r in agg)
print("empty assistant spans:", bad, "rows:", rows)
```

### 2.3 Token length distribution vs max_length

- [ ] Inspect token length distribution under your intended `--max-length`.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

DST = "/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl"
tok = AutoTokenizer.from_pretrained("/fsx/coreweave-training/release/i60")
ds = load_dataset("json", data_files=DST)["train"]

lengths = ds.map(lambda b: {"len": [len(tok(x).input_ids) for x in b["text"]]},
                 batched=True, remove_columns=ds.column_names)
lens = []
for r in lengths: lens.extend(r["len"])
lens_sorted = sorted(lens)
def perc(p):
    return lens_sorted[int(len(lens_sorted)*p)] if lens_sorted else 0
print("p50 p90 p99 max:", perc(0.5), perc(0.9), perc(0.99), max(lens_sorted) if lens_sorted else 0)
```

### 2.4 Visual preview of loss mask

- [x] Quick preview that assistant spans (green) are masked, others red.

```python
from transformers import AutoTokenizer
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY
import torch

tok = AutoTokenizer.from_pretrained("/fsx/coreweave-training/release/i60")
tmpl = TEMPLATE_REGISTRY.get("llama3-preformatted-patient-agent")
sample = load_dataset("json", data_files=DST)["train"][0]["text"]

res = preprocess_conversations(
    tokenizer=tok,
    conversations=[sample],
    chat_template=tmpl,
    max_length=1024,
    is_preformatted=True,
)
input_ids = res["input_ids"][0].squeeze()
loss_mask = res["loss_mask"][0].squeeze()
print("loss_mask sum:", int(loss_mask.sum().item()), "len:", int(loss_mask.numel()))
```

---

## Step 3 — Save artifacts and metadata

- [ ] Save `/fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl` with only `text`.
- [ ] Record metadata:
  - [ ] Template name and literal strings (headers/EOT)
  - [ ] Tokenizer/model revision; `max_length`
  - [ ] Row count; header/EOT coverage metrics
  - [ ] Assistant payload coverage; length stats; loss_mask preview stats

---

## Next Steps (reference)

### Extract hidden states (precompute)

- [x] Use the preformatted JSONL; consistent template/tokenizer/length.

```bash
torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
  --data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --output-path /opt/dlami/nvme/i60-hidden-states \
  --model-path /fsx/coreweave-training/release/i60 \
  --chat-template llama3-preformatted-patient-agent \
  --is-preformatted \
  --max-length 4096 --tp-size 8 --batch-size 1
```

### Train offline on saved states

- [ ] Use the same template/tokenizer/length as extraction.

```bash
torchrun --nproc_per_node=8 scripts/train_eagle3_offline.py \
  --target-model-path /fsx/coreweave-training/release/i60 \
  --draft-model-config configs/llama3-405B-eagle3.json \
  --train-data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --train-hidden-states-path /opt/dlami/nvme/i60-hidden-states-masked/ \
  --chat-template llama3-preformatted-patient-agent \
  --is-preformatted \
  --output-dir /fsx/brayden/outputs/i60-lr-2e5 \
  --num-epochs 10 --draft-global-batch-size 4 --draft-micro-batch-size 1 \
  --learning-rate 2e-5 --max-length 2048 --ttt-length 5 --tp-size 8
```

#### Stable training variant (increase effective batch via accumulation)

- Rationale:
  - Increase effective batch (draft_global_batch_size=16) to smooth gradients and improve stability.
  - Keep draft_micro_batch_size=1 to avoid extra VRAM; accumulation handles the larger effective batch.
  - Keep ttt-length unchanged; adjust LR only if still unstable (e.g., reduce to 1e-5).

```bash
WANDB_API_KEY="dd8a44b7199a83dabb48e348ed3b336c2fb6eba8" \
setsid torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_eagle3_offline.py \
  --target-model-path /fsx/coreweave-training/release/i60 \
  --draft-model-config configs/llama3-405B-eagle3.json \
  --train-data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --train-hidden-states-path /opt/dlami/nvme/i60-hidden-states-masked/ \
  --output-dir /fsx/brayden/outputs/i60-fixed-mask/ \
  --num-epochs 12 \
  --draft-global-batch-size 16 \
  --draft-micro-batch-size 1 \
  --learning-rate 5e-5 \
  --max-length 2048 \
  --ttt-length 5 \
  --draft-attention-backend flex_attention \
  --tp-size 8 \
  --log-steps 1 \
  --report-to wandb \
  --wandb-project i60-405b-eagle3 \
  --wandb-name "llama405b_offline_$(date +%Y%m%d_%H%M%S)" \
  --wandb-key "$WANDB_API_KEY" \
  --chat-template llama3-preformatted-patient-agent \
  --is-preformatted \
  </dev/null > /fsx/brayden/logs_correct_masking/train_i60_405b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

Example with W&B and your current settings:

```bash
WANDB_API_KEY="dd8a44b7199a83dabb48e348ed3b336c2fb6eba8" \
setsid torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_eagle3_offline.py \
  --target-model-path /fsx/coreweave-training/release/i60 \
  --draft-model-config configs/llama3-405B-eagle3.json \
  --train-data-path /fsx/brayden/SpecForge/cache/regen-dataset/preformatted_text.jsonl \
  --train-hidden-states-path /opt/dlami/nvme/i60-hidden-states-masked/ \
  --output-dir /fsx/brayden/outputs/i60-fixed-mask/ \
  --num-epochs 12 \
  --draft-global-batch-size 4 \
  --draft-micro-batch-size 1 \
  --learning-rate 2e-5 \
  --max-length 2048 \
  --ttt-length 5 \
  --draft-attention-backend flex_attention \
  --tp-size 8 \
  --log-steps 1 \
  --report-to wandb \
  --wandb-project i60-405b-eagle3 \
  --wandb-name "llama405b_offline_$(date +%Y%m%d_%H%M%S)" \
  --wandb-key "$WANDB_API_KEY" \
  --chat-template llama3-preformatted-patient-agent \
  --is-preformatted \
  </dev/null > /fsx/brayden/logs_correct_masking/train_i60_405b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## Consistency Checklist

- [ ] Same chat template string constants across formatting, extraction, and training
- [ ] Same tokenizer/model revision
- [ ] Same `max_length` (or compatible, if you intentionally change between stages)
- [ ] Assistant spans contain non-empty payload (no header-only turns)
- [x] Loss mask sums are non-trivial in spot checks


---

## Pitfalls we hit and how we fixed them

- **Assistant header with no payload (e.g., only “Agent: ”)**
  - Symptom: Tiny loss_mask sums; only the label is green.
  - Fix: In the converter, merge header-only assistant with the next assistant payload (when it has no header) or drop the empty turn. Ensure exactly one `<|eot_id|>` between turns.

- **Wrong placement of role labels in the template**
  - Symptom: Labels (“Agent: ”/“Patient: ”) contributed to the masked region or broke span detection.
  - Fix: Register `llama3-preformatted-patient-agent` with labels embedded in headers so labels are not part of assistant/user content.

- **Missing `<|eot_id|>` between turns**
  - Symptom: Assistant spans not detected, masks near zero.
  - Fix: Ensure one `<|eot_id|>` between every two turns (final EOT optional). Converter adds EOT when missing; avoids double EOTs on user turns that already contain it.

- **Auto-deriving text from conversations in prepare_hidden_states**
  - Symptom: Derived “text” from the first user message only, producing wrong masks (e.g., only “Agent:” green).
  - Fix: Remove auto-derive fallback. Always pass a proper preformatted `text` column via `--is-preformatted`.

- **Duplicate `--chat-template` definition crash**
  - Symptom: `argparse.ArgumentError: conflicting option string: --chat-template` in prepare_hidden_states.
  - Fix: Do not redefine `--chat-template` in prepare_hidden_states; it is already provided by sglang ServerArgs. Keep passing `--chat-template` on the command line.

- **Template/length/tokenizer inconsistency between stages**
  - Symptom: Misaligned masks/vocab-mapping vs hidden states; degraded training.
  - Fix: Use the same chat template, tokenizer/model, and `max_length` for conversion, extraction, and offline training.

- **Regex fragility**
  - Symptom: Span detection brittle on formatting edge cases.
  - Fix: Preferred deterministic detection: split by `<|eot_id|>`, use `startswith(assistant_header)` to mark assistant spans, map char spans to tokens using offsets.

- **System prompt handling**
  - Symptom: Concern about masking system text.
  - Fix: System text is outside assistant spans and remains unmasked (red); no fixed system prompt is required in the template.


