
# CS781 Capstone: In-House Legal Intelligence System

Rajkumar Ahirwar, Roll No. 240836, IIT Kanpur (rajkumar24@iitk.ac.in)
Codabench username: `r_240836`

A self-hosted legal NLP stack built in seven phases, so that confidential case
files never leave the firm's hardware. Phase 1 pretrains a legal language model
from scratch (four architectures, 51-62M parameters at Tier 1, a 101.9M
decoder-only model at Tier 2). Phases 2-7 each fine-tune a descendant of a
Phase-1 checkpoint for one legal task: NER, rhetorical roles, judgment
prediction, statute identification, summarization and prior-case retrieval.
No pretrained language-model weights and no external legal data are used.

This repository contains code only. It does not contain datasets, model weights
or checkpoints (see "Data and checkpoints").

## Results (Codabench, ranks as of 9 November 2026)

| Phase | Task                   | Primary metric       | Best score | Rank | Valid submissions |
| ----- | ---------------------- | -------------------- | ---------- | ---- | ----------------- |
| 1     | Legal LM pretraining   | BERTScore F1         | 0.8278     | 104  | 6 (+1 failed)     |
| 2     | Legal NER              | strict span macro-F1 | 0.6022     | 19   | 6                 |
| 3     | Rhetorical roles       | macro-F1             | 0.5063     | 57   | 7                 |
| 4     | Judgment prediction    | macro-F1             | 0.6851     | 89   | 8                 |
| 5     | Statute identification | macro-F1             | 0.4380     | 14   | 9                 |
| 6     | Summarization          | BERTScore F1         | 0.8278     | 101  | 7                 |
| 7     | Prior case retrieval   | Micro-F1@7           | 0.4808     | 14   | 11                |

Total valid submissions: 54. The full report is in `report/report.pdf`.

## Repository layout

```
cs781-capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── report/                      report.pdf, report.tex, params.pdf, subs.pdf
├── phase1_legal_lm/             phase1.ipynb, config.py, models.py, tokenizer_train.py,
│                                prepare_data.py, data_utils.py, train.py, generate.py,
│                                eval.py, distill.py
├── phase2_ner/                  phase2.ipynb, config.py, models.py, lner.py
├── phase3_rhetorical_roles/     phase3.ipynb, config.py, models.py
├── phase4_judgment_prediction/  phase4.ipynb, p1_models.py, cjpe_prepare.py,
│                                cjpe_lib.py, cjpe_train.py
├── phase5_statute_identification/  phase5.ipynb, config.py, models.py
├── phase6_summarization/        phase6.ipynb, p6text.py, p6lib.py, run_exp.py, infer.py
└── phase7_prior_case_retrieval/ phase7.ipynb, pcr_lib.py, pcr_prepare.py,
                                 pcr_train.py, boost_lib.py
```

Each phase folder is self-contained. The notebook is the entry point: it writes
the helper `.py` files next to itself and runs the experiments as resumable
subprocesses.

## Environment

All experiments ran on Kaggle (T4 or P100 GPU, 12-hour sessions, fp16 autocast
with GradScaler). Install dependencies with `pip install -r requirements.txt`.
Phases 1, 4, 6 and 7 install a few packages in their first cell, so enable
Internet in the Kaggle session.

## Data and checkpoints

The datasets are the course files and are not redistributed here. The notebooks
search `/kaggle/input` recursively, so add each dataset as a Kaggle Input with
these filenames:

| Phase | Required input files                                                                                                                                                |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `train_acts.json`, `train_high_courts.json`, `train_supreme_court.json`, `train_tribunals.json`, and the test file (`test.json` or `Phase1- Test.json`) |
| 2     | `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`, `labels.txt`                                             |
| 3     | `CL_train.json`, `IT_train.json`, `CL_dev.json`, `IT_dev.json`, `CL_test.json`, `IT_test.json`, `label_vocab.json`                                    |
| 4     | `train.json`, `dev.json`, `test.json`                                                                                                                         |
| 5     | `Train/train.json`, `Train/dev.json`, `Test/test.json`                                                                                                        |
| 6     | `train.json` (fields `id`, `document`, `summary`), `test.json`                                                                                            |
| 7     | `train.json`, `train_candidates.json`, `dev.json`, `dev_candidates.json`, `test.json`, `test_candidates.json`                                           |

Phases 2-7 also need your own Phase-1 outputs as a Kaggle Input:
`artifacts/tokenizer.json`,
`artifacts/ckpt_tier1/{decoder_only,efficient,chunk_recurrent,enc_dec}_best.pt`
and `artifacts/ckpt_tier2/decoder_only_best.pt`. Phase 7 also copies
`models.py` and `config.py` from the Phase-1 output folder. Run Phase 1 first to
produce these; they are not included in this repository.

Several phases use a file called `train.json`, so add one phase's dataset at a
time, or use the override variables listed in each phase section.

## How to run

1. Run `phase1_legal_lm/phase1.ipynb` end to end. It trains the tokenizer,
   prepares data, trains the four architectures and the Tier-2 model, and
   writes the Phase-1 prediction file. Save its `artifacts/` folder as a Kaggle
   dataset.
2. For each phase 2-7, add that phase's dataset and the Phase-1 dataset as
   Inputs, open the notebook and run it top to bottom. Submissions are written
   under `artifacts/submissions/` (Phase 5 uses `lsi_artifacts/submissions/`) as
   a `prediction.json` and a zip file in the Codabench format.
3. Experiments are subprocesses that write checkpoints and score files, and
   finished experiments are skipped after a session restart. Set the `FORCE*`
   flag in a notebook to retrain.

## Shared design

- Tokenizer: byte-level BPE, 32k vocabulary, trained on a sample of all four
  corpus sources. All 256 bytes are in the alphabet, text is not lowercased.
  Special ids: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3.
- Backbone family: RMSNorm pre-norm, SwiGLU feed-forward (hidden multiplier
  2.75), tied input/output embeddings.
- Adaptation techniques implemented from scratch: LoRA, rsLoRA, LoRMA
  (multiplicative low-rank update), Houlsby and Pfeiffer adapters, BitFit, IA3,
  prompt tuning, partial and full fine-tuning, knowledge distillation (KD).

## Model architecture

| Architecture    | Key components                                                                                          | Tier-1 size               |
| --------------- | ------------------------------------------------------------------------------------------------------- | ------------------------- |
| decoder-only    | RoPE, grouped-query attention (GQA), causal mask                                                        | 57.4M (d=640, 8 layers)   |
| efficient       | local sliding-window attention (128), full attention every 4th block                                    | 57.4M                     |
| encoder-decoder | learned positions, bidirectional encoder, cross-attention decoder                                       | 61.6M (d=512, 6+6 layers) |
| chunk-recurrent | GRU compresses 128-token chunks into memory; the decoder cross-attends to memory of earlier chunks only | 51.3M                     |

Tier-2 model (used for Phase 1): decoder-only, d=768, 12 layers, 12 heads, 4 KV
heads, 101.9M parameters (tied embedding 24.6M plus 12 blocks of 6.44M),
context 1024.

How phases 2-7 use the backbone:

| Phase | Attention mask                   | Techniques compared                                                 |
| ----- | -------------------------------- | ------------------------------------------------------------------- |
| 2     | bidirectional                    | LoRMA, full FT, KD                                                  |
| 3     | bidirectional (causal ablated)   | joint vs per-domain, linear vs BiLSTM head, KD, random-init control |
| 4     | causal, mean-pool                | E1 LoRMA, E2 adapters, E3 BitFit, E4 KD                             |
| 5     | causal (enc-dec: native encoder) | LoRMA, partial FT, adapters, distilled full FT                      |
| 6     | causal generator                 | LoRA, rsLoRA, Pfeiffer, LoRMA variants, KD                          |
| 7     | causal, position-weighted mean   | LoRMA, IA3, last-two-block FT, full FT, KD                          |

## Implementation details

- PEFT methods are implemented from scratch. LoRMA is a multiplicative update
  `W' = (I + sBA)W`, computed as `y = h + sB(Ah)` with `h = Wx` and `B = 0` at
  initialisation.
- Phase 6 replaces `nn.MultiheadAttention` with a module using separate
  q/k/v/out linear layers so adapters reach `out_proj`, and checks that logits
  are unchanged (max difference about 1e-5).
- KD loss is `alpha * T^2 * KL + (1 - alpha) * CE`, plus a hidden-state cosine
  term in Phases 2 and 3. Student models differ by phase: P2 copies layers from
  the fine-tuned teacher, P4 uses a half-depth chunk-recurrent student, P5
  distils into the unpruned 6-layer encoder-decoder, P7 prunes to one third of
  the depth.
- Training data is read through `uint16` memmap shards to avoid out-of-memory
  kills on the 19 GB corpus. Long runs are subprocesses that write checkpoints
  and score files.

## Phase 1: Legal language model

Folder `phase1_legal_lm/`, entry point `phase1.ipynb`.

- Data: the four corpora (about 19 GB) are streamed with `ijson`, up to 250M
  tokens kept per source, split by document (2% validation). Judgments are
  up-weighted (1.5 for High Court and Supreme Court, 1.0 for the others).
- Tier 1: decoder-only 57.4M, efficient 57.4M, encoder-decoder 61.6M,
  chunk-recurrent 51.3M, each trained on the same data and budget (25M tokens,
  1,500 steps) so the comparison reflects the architecture.
- Tier 2: decoder-only, 101.9M parameters, context 1024.
- Training: next-token cross-entropy, AdamW (betas 0.9/0.95, weight decay 0.1),
  peak learning rate 3e-4 with cosine decay to 10%, batch 8 with 4 accumulation
  steps, 5,000 planned steps (about 165M tokens), fp16.
- Generation: nucleus and top-k sampling with repetition penalty and 4-gram
  blocking, trimmed to 128 words. Settings chosen from a three-setting sweep by
  BERTScore.
- Evaluation: held-out (512-word prompt, 128-word reference) pairs scored by
  BERTScore, ROUGE-L and BLEU-4 (`eval.py`).
- Fixed bugs: out-of-memory kills from loading all JSON in every process, a
  label off-by-one, a future leak in the chunk-recurrent memory, and an `<eos>`
  appended to the prompt.
- Best submission: Tier-2 `decoder_only_best.pt` through `generate.py`.
  [Fill in the exact sampling settings.]

## Phase 2: Legal NER

Folder `phase2_ner/`, entry point `phase2.ipynb`. Needs all four Phase-1
checkpoints and the tokenizer.

- Task: 12 entity types, strict character-span macro-F1.
- Method: BIO token classification with exact character-offset reconstruction.
  A gold-to-BIO-to-spans round trip must reach macro-F1 of at least 0.97 before
  training.
- Decoding: per-token probabilities are averaged over overlapping windows (edge
  tokens down-weighted), the document is decoded once with constrained BIO
  Viterbi, and an O-bias trades precision against recall. Boundary punctuation
  is stripped by per-label rules learned from training gold.
- Training: mild class weights, differential learning rates (backbone 6e-5, head
  1e-3), optional BiLSTM head and span-boundary head. Top models are refit on
  all 70 labelled documents.
- Best submission: 0.6022. [Fill in the experiment tag and settings.]

## Phase 3: Rhetorical roles

Folder `phase3_rhetorical_roles/`, entry point `phase3.ipynb`.

- Task: one of 13 roles per sentence for Corporate Law and Information
  Technology documents, macro-F1.
- Method: the sentence vector concatenates the mean of its token states and the
  state at its `<eos>`, plus relative-position features and a domain embedding,
  followed by a linear or BiLSTM head over the sentences of a window. Sentences
  are truncated to 64 tokens and packed into 512-token windows.
- Decoding: overlapping windows with edge-weighted probability averaging, and
  optional Viterbi smoothing with a transition matrix from training labels.
- Training: one joint CL+IT model, with CL-only and IT-only models as a study.
  Class weights are the square root of inverse frequency, clipped at 10. Final
  models are refit on train+dev with two seeds and averaged.
- Best submission: 0.5063. [Fill in the experiment tag and settings.]

## Phase 4: Judgment prediction

Folder `phase4_judgment_prediction/`, entry point `phase4.ipynb`. To pin
specific checkpoints, set `CKPT_OVERRIDE`.

- Task: binary court judgment prediction, macro-F1.
- Method: a hierarchical classifier encodes up to four 512-token windows per
  document with the Phase-1 backbone, mean-pools tokens, adds a relative-position
  embedding, aggregates with attention and applies an MLP head. Documents longer
  than 16,384 tokens keep 25% head and 75% tail. Training uses a stratified
  5,000-document subsample.
- Required experiments: E1 LoRMA on decoder-only (with an extra diagonal scale),
  E2 adapters on efficient, E3 BitFit on encoder-decoder, E4 distillation into a
  half-depth chunk-recurrent student.
- Decision: class-weighted cross-entropy and a decision threshold tuned on dev.
- Best submission: 0.6851. [Fill in the experiment tag and settings.]

## Phase 5: Statute identification

Folder `phase5_statute_identification/`, entry point `phase5.ipynb`. Works with
any subset of the Phase-1 checkpoints. Set `QUICK` for a short run. Optional
overrides: `DATA_DIR_OVERRIDE`, `TOKENIZER_OVERRIDE`, `CKPT_OVERRIDE`.

- Task: multi-label identification of about 100 statutes, macro-F1.
- Method: chunks are pooled by mean and max; document logits combine gated
  attention over chunks, max over chunks, and a label-wise attention term. Loss
  is BCE with a positive weight, or asymmetric loss.
- Thresholds: a global value shrunk towards per-label values, with the shrinkage
  chosen by split-half cross-validation on dev.
- Version 2: a longer EMA-averaged distilled encoder-decoder encoder,
  temperature-calibrated dev-weighted ensembling, and optional hashed 1-3-gram
  TF-IDF "wide" models. The wide models are not Phase-1 descendants, and the
  ensemble weights use dev labels, so local dev scores for v2 are optimistic.
- Best submission: 0.4380. [State whether it includes the wide models, and give
  the experiment tag.]

## Phase 6: Summarization

Folder `phase6_summarization/`, entry point `phase6.ipynb`. Needs all four
Phase-1 checkpoints (`{arch}_best.pt`). Optional overrides: `TRAIN_OVERRIDE`,
`TEST_OVERRIDE`, `TOKENIZER_OVERRIDE`, `P1_CKPT_OVERRIDE`.

- Task: abstractive summarization of long judgments, BERTScore primary.
- Method: decoder-style models see `[bos] doc "Summary:" summary [eos]` with loss
  on the summary only; the encoder-decoder encodes the document and decodes the
  summary. There is no dev split, so 200 training documents are held out (50
  are generated and scored).
- Long documents: truncation, head+tail, importance-based sentence selection,
  and hierarchical chunk summaries.
- Decoding: our own beam search with repetition penalty, n-gram blocking, length
  control and an optional soft copy bias.
- Adaptation: LoRA, rsLoRA, Pfeiffer adapters and LoRMA (plain,
  permutation-inflated and additive variants), plus cross-architecture
  distillation.
- Best submission: 0.8278. [Fill in the experiment name and decoding settings.]

## Phase 7: Prior case retrieval

Folder `phase7_prior_case_retrieval/`, entry point `phase7.ipynb`. Needs
`decoder_only_best.pt` or `efficient_best.pt`, the tokenizer, and `models.py`
and `config.py` from the Phase-1 output. Optional overrides: `P1_CKPT`,
`P1_TOKENIZER`.

- Task: return seven prior cases per query from the split's own candidate pool,
  Micro-F1@7.
- Dense model: a bi-encoder over the Phase-1 stack with position-weighted mean
  pooling, scored by document mean and by MaxSim over up to 10 chunks. Training
  is InfoNCE (temperature 0.05) with in-batch and TF-IDF hard negatives.
- Adaptations compared: LoRMA, IA3, last-two-block fine-tuning, full
  fine-tuning, and distillation into a student at one third of the depth.
- Final system: each dense score is fused with nine lexical scorers (TF-IDF
  uni-gram and bi-gram, BM25, citation fingerprint, rare terms, head, tail, LSA,
  5-gram shingles) by a two-stage gradient-boosted ranker trained on mini-worlds
  built from the train labels, then Sinkhorn normalisation and pseudo-relevance
  feedback tuned on dev.
- Caveats: BM25 parameters were tuned on train+dev, and Sinkhorn normalisation
  is transductive over the split's score matrix.
- Best submission: 0.4808. [Fill in the experiment name.]

## Submissions

Every valid Codabench submission per phase, in the order submitted. The best is
in bold. Use the experiment tag in the last column to find the code that
produced it.

### Phase 1 (6 valid, 1 failed)

| #   | Date   | BERTScore F1         | Model / decoding settings | Tag    |
| --- | ------ | -------------------- | ------------------------- | ------ |
| 1-6 | [fill] | [fill] (best 0.8278) | [fill]                    | [fill] |

### Phase 2 (6 valid)

| # | Score            | Variant                                                                       |
| - | ---------------- | ----------------------------------------------------------------------------- |
| 1 | 0.4075           | v1: union of per-window spans (validation 0.373, precision 0.28, recall 0.52) |
| 2 | 0.5369           | v2 [fill]                                                                     |
| 3 | 0.5545           | v2 [fill]                                                                     |
| 4 | 0.5510           | v2 [fill]                                                                     |
| 5 | **0.6022** | v2: probability averaging + Viterbi + O-bias [fill tag] (rank 19)             |
| 6 | 0.4285           | v2 [fill]                                                                     |

### Phase 3 (7 valid)

| # | Score            | Variant              |
| - | ---------------- | -------------------- |
| 1 | 0.4718           | [fill]               |
| 2 | 0.4645           | [fill]               |
| 3 | 0.3908           | [fill]               |
| 4 | 0.4645           | [fill]               |
| 5 | 0.5048           | [fill]               |
| 6 | **0.5063** | [fill tag] (rank 57) |
| 7 | 0.4662           | [fill]               |

### Phase 4 (8 valid)

| # | Score            | Variant              |
| - | ---------------- | -------------------- |
| 1 | 0.6368           | [fill]               |
| 2 | 0.5429           | [fill]               |
| 3 | 0.5429           | [fill]               |
| 4 | **0.6851** | [fill tag] (rank 89) |
| 5 | 0.6092           | [fill]               |
| 6 | 0.5681           | [fill]               |
| 7 | 0.5960           | [fill]               |
| 8 | 0.5409           | [fill]               |

### Phase 5 (9 valid)

| #   | Date   | Score                                                  | Variant                                                  |
| --- | ------ | ------------------------------------------------------ | -------------------------------------------------------- |
| 1-7 | 28 Oct | 0.3198, 0.2560, 0.2614, 0.2616, 0.2991, 0.3198, 0.2991 | v1 (distilled enc-dec full FT: 0.3198) [fill the others] |
| 8   | 9 Nov  | 0.4248                                                 | v2 [fill]                                                |
| 9   | 9 Nov  | **0.4380**                                       | v2 [fill tag] (rank 14)                                  |

### Phase 6 (7 valid)

| # | BERTScore F1     | Variant               |
| - | ---------------- | --------------------- |
| 1 | 0.8231           | [fill]                |
| 2 | 0.8223           | [fill]                |
| 3 | 0.8217           | [fill]                |
| 4 | 0.8140           | [fill]                |
| 5 | 0.8259           | [fill]                |
| 6 | 0.8263           | [fill]                |
| 7 | **0.8278** | [fill tag] (rank 101) |

### Phase 7 (11 valid)

| #    | Date  | Micro-F1@7                                       | Variant                      |
| ---- | ----- | ------------------------------------------------ | ---------------------------- |
| 1-6  | 7 Nov | 0.4223, 0.4229, 0.4217, 0.4259, 0.4259, 0.4235   | [fill]                       |
| 7-11 | 8 Nov | 0.4790, 0.4796, **0.4808**, 0.4784, 0.4755 | [fill] (best is #9, rank 14) |

## Notes and limitations

- Single seeds, small dev sets (14 documents in Phase 2, 118 queries in Phase 7)
  and a small Tier-2 token budget (about 165M tokens).
- Differences between close variants are within noise; we judged them on
  Codabench.
- The metrics measure similarity to a reference, not legal correctness.

## LLM use

Claude (Anthropic) was used for clarifying concepts, debugging help, figure code
and grammar and wording of the report. See the LLM Usage Declaration in
`report/report.pdf`.
