# StoryReasoning: Context-to-Next-Text Prediction with Memory

This project builds and evaluates a model that predicts the next story frame's text using the previous K frames (text + images). It compares a standard baseline against a memory-augmented model to test whether explicit memory improves long-range entity consistency.

## Objective

Build and evaluate a model that predicts the next story frame's text using the previous K frames. Compare a baseline model against a memory-slot model to assess whether explicit memory improves long-range entity consistency.

## Dataset

Source: Hugging Face dataset `daniel3303/StoryReasoning`.

Dataset splits (as provided):

| Split | Stories |
| --- | --- |
| Train | 3552 |
| Test | 626 |

Each story contains:

- `story_id` (string)
- `images` (list of frames as images)
- `frame_count` (number of frames)
- `story` (text containing per-frame segments and entity tags)
- `chain_of_thought` (not used)

## Preprocessing and Data Pipeline

### Train/Val split

- Story-level split on the train split to avoid leakage.
- Train stories: 3196
- Val stories: 356
- Test stories: 626 (official test split)

### Sliding window sample generation

Goal: predict target frame `t` using the previous `K` frames as context.

- Context length `K = 6`
- For each story with `frame_count >= K + 1`, create samples for `t` in `[K, frame_count - 1]`.

Per-sample outputs:

- `ctx_images`: frames `[t-K ... t-1]` (K images)
- `ctx_texts`: text segments `[t-K ... t-1]` (K text chunks)
- `target_text`: text chunk at frame `t`
- `ctx_entity_sets`: entity IDs parsed per context frame
- `target_entity_set`: entity IDs parsed for target frame

### Tokenization

- Tokenizer: `distilbert-base-uncased`
- Context text max length per frame: 128 tokens
- Target text max length: 160 tokens

Batching shapes verified:

| Tensor | Shape |
| --- | --- |
| `ctx_input_ids` | `[B, K, 128]` |
| `tgt_input_ids` | `[B, 160]` |

### Image embeddings

- Backbone: pretrained ResNet18 (torchvision)
- Feature extractor frozen (fc replaced with identity)
- Each image embedded to 512-d vector

Embedding shapes verified:

| Tensor | Shape |
| --- | --- |
| `ctx_img_emb` | `[B, K, 512]` |
| `tgt_img_emb` | `[B, 512]` |

Note: target image embedding is reserved for later extensions and is not required for text prediction.

## Exploratory Data Analysis (EDA)

### Entity continuity motivation

To quantify whether target entities are usually present in the context, the following was computed:

`GT_support = |target_entities ∩ union(context_entities)| / |target_entities|`

Method:

- Parse entity IDs from `<gdo ...>` tags (deduplicate per frame).

Result (val subset):

- `GT_support ≈ 0.7241`

Interpretation: about 72% of target entities appear in the previous context, indicating meaningful continuity and a plausible benefit from memory mechanisms.

### Entity gap statistics (deduplicated)

After fixing within-frame duplicates:

| Metric | Value |
| --- | --- |
| Mean gap | 1.94 frames |
| Median gap | 1 |
| Max gap | 20 |

Interpretation: entities frequently re-appear within a few frames, with some longer gaps.

## Models

Both models perform next-text prediction via teacher forcing, conditioned on a context summary.

### Shared components

- Frame-level text encoder: token embedding + masked mean pooling -> one vector per frame.
- Fusion: concatenate (frame_text_vec, frame_image_emb) then project to hidden size.
- Context encoder: GRU over K fused frame vectors -> final hidden state `h0`.
- Decoder: GRU language model conditioned on `h0`, predicting next tokens for `target_text`.

### Baseline model

- No explicit memory; context information is captured only by the context GRU.

### Memory-augmented model

- Adds `M = 16` learnable memory slots with attention read + gated write per timestep.
- At each context timestep, reads memory using the fused frame vector, updates memory, and feeds (frame + memory_readout) to the context GRU.

## Training setup

- Device: CUDA GPU
- Batch size: 8
- Optimizer: AdamW (`lr = 3e-4`, `weight_decay = 0.01`)
- Epochs run: 2
- Validation evaluated on a capped subset for speed (120 batches)

Metrics logged:

- Cross-entropy loss (teacher forcing)
- Perplexity (exp(loss))
- Entity-tag generation metrics (attempted; see Results)

## Results

### Baseline (epoch 2 best checkpoint)

| Metric | Value |
| --- | --- |
| Val loss | 1.3463 |
| Val perplexity | 3.84 |
| Entity-tag generation rate | 0.000 |

### Memory model (epoch 2 best checkpoint)

| Metric | Value |
| --- | --- |
| Val loss | 1.3555 |
| Val perplexity | 3.88 |
| Entity-tag generation rate | 0.000 |

### Entity consistency / hallucination metrics (attempted)

Entity-tag-based generation metrics relied on the model producing tags like `<gdo char1>...</gdo>`. The decoder did not emit tag structures, so:

- `GEN_tag_rate = 0.000` for both baseline and memory on evaluated subsets.
- `GEN_new_entity_rate`, `GEN_entity_precision`, `GEN_entity_recall`, and `GEN_entity_F1` were all 0.0 and not meaningful.

## Findings and interpretation

- The task has genuine entity continuity (`GT_support ≈ 0.724`), so memory mechanisms are theoretically relevant.
- With the current architecture and training budget (2 epochs), the memory-slot model did not improve perplexity; the baseline performed slightly better (about 0.7% lower val loss).
- Tag-based hallucination metrics require structured tag emission; since the model does not generate `<gdo ...>` tags, the metrics are not sensitive in this setup.

## Limitations

- Training budget was small (2 epochs), which may be insufficient for memory advantages to emerge.
- Decoder objective does not explicitly enforce tag reproduction; tag-based metrics cannot be used directly.
- The decoder produces mostly natural-language tokens; entity tags are not preserved.

## Future work

- Train longer (5 to 10 epochs) and compare again.
- Use a tokenizer/model that preserves markup more reliably (for example, byte-level BPE such as GPT-2 tokenizer).
- Add a structured objective that forces tag reproduction (for example, constrained decoding or a tag-aware LM loss).
- Replace or augment tag-based metrics with name-based consistency checks (extract surface names from GT tags and check if generated text mentions those names).

## Repository layout

- `src/dataloader.py`: frame parsing, entity extraction, windowed dataset, and tokenizer collation.
- `src/encoders_image.py`: ResNet18 embedding utilities.
- `src/encoders_text.py`: tokenizer loading helpers.
- `src/model_baseline.py`: baseline next-text model.
- `src/memory_module.py`: memory slots with read/write operations.
- `src/model_memory.py`: memory-augmented next-text model.
- `src/train.py`: training loop, decoding, and hallucination evaluation.
- `src/eval_longseq.py`: evaluation helpers (loss, perplexity, hallucination).

## Quickstart

```bash
cd "Vishalini"

# Train memory model
python src/train.py --model memory

# Train baseline
python src/train.py --model baseline
```
