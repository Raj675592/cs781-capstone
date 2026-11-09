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
or checkpoints (see "Data and checkpoints"). The Phase-1 tokenizer is produced by `phase1/phase1.ipynb`.

## Results (Codabench, ranks as of 9 November 2026)

| Phase | Task | Primary metric | Best score | Rank | Valid submissions |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | 0.8278 | 104 | 6 (+1 failed) |
| 2 | Legal NER | strict span macro-F1 | 0.6022 | 19 | 6 |
| 3 | Rhetorical roles | macro-F1 | 0.5063 | 57 | 7 |
| 4 | Judgment prediction | macro-F1 | 0.6851 | 89 | 8 |
| 5 | Statute identification | macro-F1 | 0.4380 | 14 | 9 |
| 6 | Summarization | BERTScore F1 | 0.8278 | 101 | 7 |
| 7 | Prior case retrieval | Micro-F1@7 | 0.4808 | 14 | 11 |

Total valid submissions: 54.

## Repository layout

```
cs781-capstone/
├── README.md
├── requirements.txt
├── .gitignore
├── phase1/   phase1.ipynb, submission.zip   legal language model pretraining
├── phase2/   phase2.ipynb, submission.zip   legal NER
├── phase3/   phase3.ipynb, submission.zip   rhetorical roles
├── phase4/   phase4.ipynb, submission.zip   judgment prediction
├── phase5/   phase5.ipynb, submission.zip   statute identification
├── phase6/   phase6.ipynb, submission.zip   summarization
└── phase7/   phase7.ipynb, submission.zip   prior case retrieval
```

Each phase folder contains one notebook and one submission file. The notebook is
the entry point and contains all source code: when it runs, it writes its helper
`.py` files (model, config, training, evaluation and decoding code) with
`%%writefile` and runs the experiments as resumable subprocesses.
`submission.zip` is the file that gave our best Codabench score for that phase
(the score is in the results table below). It holds predictions only, not model
weights.

## Environment

All experiments ran on Kaggle (T4 or P100 GPU, 12-hour sessions, fp16 autocast
with GradScaler). Install dependencies with `pip install -r requirements.txt`.
Phases 1, 4, 6 and 7 install a few packages in their first cell, so enable
Internet in the Kaggle session.

## Data and checkpoints

The datasets are the course files and are not redistributed here. The notebooks
search `/kaggle/input` recursively, so add each dataset as a Kaggle Input with
these filenames:

| Phase | Required input files |
|---|---|
| 1 | `train_acts.json`, `train_high_courts.json`, `train_supreme_court.json`, `train_tribunals.json`, and the test file (`test.json` or `Phase1- Test.json`) |
| 2 | `fold1-text.json`, `fold1-gold.json`, `fold2-text.json`, `fold2-gold.json`, `fold3-text.json`, `labels.txt` |
| 3 | `CL_train.json`, `IT_train.json`, `CL_dev.json`, `IT_dev.json`, `CL_test.json`, `IT_test.json`, `label_vocab.json` |
| 4 | `train.json`, `dev.json`, `test.json` |
| 5 | `Train/train.json`, `Train/dev.json`, `Test/test.json` |
| 6 | `train.json` (fields `id`, `document`, `summary`), `test.json` |
| 7 | `train.json`, `train_candidates.json`, `dev.json`, `dev_candidates.json`, `test.json`, `test_candidates.json` |

Phases 2-7 also need your own Phase-1 outputs as a Kaggle Input:
`artifacts/tokenizer.json`,
`artifacts/ckpt_tier1/{decoder_only,efficient,chunk_recurrent,enc_dec}_best.pt`
and `artifacts/ckpt_tier2/decoder_only_best.pt`. Phase 7 also copies
`models.py` and `config.py` from the Phase-1 output folder (the Phase-1 notebook
writes them). The checkpoints are model weights and are not included in this
repository: run `phase1/phase1.ipynb` first to produce them, then save
its `artifacts/` folder as a Kaggle dataset. The tokenizer (`tokenizer.json`) is also
produced by that notebook.

Several phases use a file called `train.json`, so add one phase's dataset at a
time, or use the override variables listed in each phase section.

## How to run

1. Run `phase1/phase1.ipynb` end to end. It trains the tokenizer,
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

## Experimental setup

**Compute.** Kaggle T4 or P100 GPUs, 12-hour sessions, fp16 autocast with
GradScaler (bf16 on Ampere or newer). Each experiment is a subprocess that
writes checkpoints and score files, so finished work is skipped after a session
kill. All runs use AdamW with warmup, cosine decay and gradient clipping at 1.0.

**Typical budgets.** Tier-1 pretraining 1,500 steps; Tier-2 up to 180 minutes
(5,000 planned steps); P2 2,400 steps; P3 2,500 steps; P4 up to 60 minutes per
main run (3 epochs); P5 6 epochs; P6 800 steps (about 2 epochs, 18-28 minutes);
P7 800 steps (60-minute cap).

**Hyperparameters of the final configurations.**

| | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|---|
| Parameters (M) | 101.9 | 57-102 | 57-102 | 51-102 | 51-102 | 51-102 | 57-102 |
| Context | 1024 | 512 | 512 | 512 x 4 windows | 512 x 8 windows | 1024 | 512 x 10 chunks |
| Peak LR | 3e-4 | 6e-5 backbone / 1e-3 head | 5e-5 (+ head) | 5e-5 to 1e-3 | 1e-4 to 6e-4 | 5e-5 to 2e-4 | 2e-5 to 3e-3 |
| Batch x accumulation | 8 x 4 | 4 | 8 | 2 x 8 | 4 x 4 | 4 x 4 | 8 queries |
| Steps / epochs | 5000 | 2400 | 2500 | 3 ep | 6 ep | 800 | 800 |
| Dropout | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.05 | 0 |
| Decoding / thresholds | top-p 0.9 | O-bias, Viterbi | Viterbi lambda | tuned threshold | tuned thresholds | beam 4 | Sinkhorn, PRF |

Learning-rate ranges span full fine-tuning (low) to PEFT, BitFit and IA3 (high).

**Model selection.** Validation loss (P1), strict span macro-F1 (P2), dev macro-F1
(P3-P5), validation loss then BERTScore (P6), dev F1@7 (P7).

**Local evaluation.**
- P1: 2% document-level held-out text and a fixed equal-weight validation set,
  plus held-out (512-word prompt, 128-word reference) pairs scored by BERTScore,
  ROUGE-L and BLEU-4.
- P2: 20% document split (14 documents).
- P6: 200 held-out training documents, with generation scored on 50.
- P7: 118 dev queries, so differences of 0.01-0.02 are within noise.

Local and leaderboard scores diverged most in P2 (0.373 vs 0.4075 for v1) and P7,
so close variants were judged by Codabench.

**Ablations and controlled comparisons.** One factor changed at a time:
tokenizer vocabulary (16k / 32k / 50k); data mix (uniform vs judgment-heavy);
context length 512 vs 1024; regularisation (weight decay 0.1 vs 0.3); best vs
final checkpoint; decoding sweeps; P3 context ablations (sentence-only, window,
BiLSTM, causal vs bidirectional, position features, class balancing, random
init); P4 representations (head, tail, head+tail with mean, attention or
transformer pooling) and a section study; P6 length, repetition and copy-bias
sweeps and a random-init control; P7 per-stage gains from lexical scorers, the
learned ranker, dense scores, Sinkhorn and PRF.

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

| Architecture | Key components | Tier-1 size |
|---|---|---|
| decoder-only | RoPE, grouped-query attention (GQA), causal mask | 57.4M (d=640, 8 layers) |
| efficient | local sliding-window attention (128), full attention every 4th block | 57.4M |
| encoder-decoder | learned positions, bidirectional encoder, cross-attention decoder | 61.6M (d=512, 6+6 layers) |
| chunk-recurrent | GRU compresses 128-token chunks into memory; the decoder cross-attends to memory of earlier chunks only | 51.3M |

Tier-2 model (used for Phase 1): decoder-only, d=768, 12 layers, 12 heads, 4 KV
heads, 101.9M parameters (tied embedding 24.6M plus 12 blocks of 6.44M),
context 1024.

How phases 2-7 use the backbone:

| Phase | Attention mask | Techniques compared |
|---|---|---|
| 2 | bidirectional | LoRMA, full FT, KD |
| 3 | bidirectional (causal ablated) | joint vs per-domain, linear vs BiLSTM head, KD, random-init control |
| 4 | causal, mean-pool | E1 LoRMA, E2 adapters, E3 BitFit, E4 KD |
| 5 | causal (enc-dec: native encoder) | LoRMA, partial FT, adapters, distilled full FT |
| 6 | causal generator | LoRA, rsLoRA, Pfeiffer, LoRMA variants, KD |
| 7 | causal, position-weighted mean | LoRMA, IA3, last-two-block FT, full FT, KD |

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

Folder `phase1/`, entry point `phase1.ipynb`.

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
- Submission: `generate.py` writes `prediction.json` and `submission.zip` from a
  Phase-1 checkpoint (Tier-2 `decoder_only_best.pt`, `_final.pt`, or the
  distilled student from `distill.py`), with sampling settings chosen in the
  decoding sweep cell.

## Phase 2: Legal NER

Folder `phase2/`, entry point `phase2.ipynb`. Needs all four Phase-1
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
- Submissions: the notebook writes `official1..4_<arch>_<technique>` (four
  different Phase-1 models; the set must contain LoRMA and KD and at least three
  techniques), `extra_span_head_decoder_only` (span-boundary head) and
  `extra_ensemble` (probability-averaged ensemble), each in `artifacts/submissions/`.
  Earlier cells also write `strict1..4_*`, `tech_<technique>_<arch>` and
  `submission_lner.zip`. Best leaderboard score 0.6022.

## Phase 3: Rhetorical roles

Folder `phase3/`, entry point `phase3.ipynb`.

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
- Submissions: the notebook selects the "strict four" (one experiment per
  Phase-1 architecture, containing LoRMA and KD, highest mean dev macro-F1),
  refits them on train+dev, and also writes an ensemble, all in
  `artifacts/submissions/<name>/` (`prediction.json` plus per-domain files).
  Best leaderboard score 0.5063.

## Phase 4: Judgment prediction

Folder `phase4/`, entry point `phase4.ipynb`. To pin
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
- Submissions: `e1_lorma_decoder_only`, `e2_adapters_efficient`,
  `e3_bitfit_enc_dec`, `e4_kd_chunk_recurrent`, written as
  `artifacts/submissions/<tag>.json` with predictions in {0,1} per test case.
  Best leaderboard score 0.6851.

## Phase 5: Statute identification

Folder `phase5/`, entry point `phase5.ipynb`. Works with
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
- Submissions: the notebook writes one submission per candidate in `CANDS`, plus
  `FINAL` (best ensemble, which may include the lexical wide models) and
  `FINAL_neural_only` (Phase-1 models only), under
  `lsi_artifacts/submissions/<name>/prediction.json`. Best leaderboard score
  0.4380.

## Phase 6: Summarization

Folder `phase6/`, entry point `phase6.ipynb`. Needs all four
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
- Submissions: for each Phase-1 architecture the best non-control experiment by
  validation BERTScore is selected, giving four files
  `artifacts/submissions/<arch>__<experiment>.json` (list of `{id, summary}`),
  and the best of them is copied to `submission_best.json`. Set `PINNED` to force
  a particular experiment. Best leaderboard score 0.8278.

## Phase 7: Prior case retrieval

Folder `phase7/`, entry point `phase7.ipynb`. Needs
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
- Submissions: `boost_exp1_lorma`, `boost_exp2_ia3`, `boost_exp3_partial`,
  `boost_exp4_full`, `boost_exp5_kd` (each fuses its own adapted dense model
  with the lexical ranker), `boost_ensemble_all_runs`, and
  `REFERENCE_lexical_only_NO_phase1_model` (a reference that uses no Phase-1
  model), under `artifacts/submissions_boost/`. Best leaderboard score 0.4808.

## Submissions

For each phase, the table gives the best valid Codabench submission. That file is
`phaseX/submission.zip`. The notebook can generate several submission files per
phase; they are listed under each table.

### Phase 1
| Best score (BERTScore F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.8278** | 104 | 6 (+1 failed) | `phase1/submission.zip` |

Generated by `generate.py` as `prediction.json` and `submission.zip` from a
Phase-1 checkpoint (Tier-2 `decoder_only_best.pt`, `_final.pt`, or the distilled
student from `distill.py`) with the sampling settings from the decoding sweep.

### Phase 2
| Best score (strict span macro-F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.6022** | 19 | 6 | `phase2/submission.zip` |

Generated: `official1..4_<arch>_<technique>`, `extra_span_head_decoder_only`,
`extra_ensemble`, `strict1..4_*`, `tech_<technique>_<arch>`, `submission_lner.zip`.
The best score comes from the v2 pipeline (probability averaging, constrained
Viterbi, O-bias).

### Phase 3
| Best score (macro-F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.5063** | 57 | 7 | `phase3/submission.zip` |

Generated: the strict four (one experiment per Phase-1 architecture), their
train+dev refits, and the ensemble.

### Phase 4
| Best score (macro-F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.6851** | 89 | 8 | `phase4/submission.zip` |

Generated: `e1_lorma_decoder_only`, `e2_adapters_efficient`, `e3_bitfit_enc_dec`,
`e4_kd_chunk_recurrent`, and re-thresholded or re-run variants.

### Phase 5
| Best score (macro-F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.4380** | 14 | 9 | `phase5/submission.zip` |

Generated: one per candidate in `CANDS`, plus `FINAL` (best ensemble, which may
include the lexical wide models) and `FINAL_neural_only` (Phase-1 models only).

### Phase 6
| Best score (BERTScore F1) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.8278** | 101 | 7 | `phase6/submission.zip` |

Generated: `<arch>__<experiment>.json` for the four architectures, and the best of
them as `submission_best.json`.

### Phase 7
| Best score (Micro-F1@7) | Rank | Valid submissions | File |
|---|---|---|---|
| **0.4808** | 14 | 11 | `phase7/submission.zip` |

Generated: `boost_exp1_lorma`, `boost_exp2_ia3`, `boost_exp3_partial`,
`boost_exp4_full`, `boost_exp5_kd`, `boost_ensemble_all_runs`.

## Notes and limitations

- Single seeds, small dev sets (14 documents in Phase 2, 118 queries in Phase 7)
  and a small Tier-2 token budget (about 165M tokens).
- Differences between close variants are within noise; we judged them on
  Codabench.
- The metrics measure similarity to a reference, not legal correctness.

## LLM use

Claude (Anthropic) was used for clarifying concepts, debugging help, figure code
and grammar and wording of the report. See the LLM Usage Declaration in the project report.
