# SequenceWeaver — Implementation Plan

**Project:** Multimodal LLM framework for coherent story generation from 2–5 unordered images
**Pipeline:** SigLIP encoding → Transformer-based ordering module → LLaVA-1.5-13B story generation (arc-aware prompting)
**Timeline:** 20 weeks (compute: Colab free/pro + local i5 Windows)
**Budget:** ~$20–43 in GPU credits

---

## 1. Guiding Principle

Only **one** component in this system needs to be trained from scratch: **the ordering module**. Everything else (SigLIP, LLaVA) is used pretrained/frozen. So the implementation should front-load effort on the ordering module and the prompting pipeline — those are your actual thesis contributions — and treat the encoder and LLM as plug-and-play components.

Build order matters. Don't build top-down (UI first). Build **bottom-up**, in this order:

1. Data → 2. Embeddings → 3. Ordering model → 4. Story generation → 5. Evaluation → 6. (Optional) thin demo UI

---

## 2. Repository Structure

Set this up on day one — it keeps local and Colab work in sync via Google Drive or GitHub.

```
sequenceweaver/
├── data/
│   ├── raw/                  # downloaded VIST, PororoSV
│   ├── processed/            # cleaned, split into train/val/test
│   └── embeddings/           # cached SigLIP vectors (.npy / .pt)
├── src/
│   ├── encoding/             # SigLIP wrapper
│   ├── ordering/             # transformer ranker: model, train, infer
│   ├── generation/           # prompt templates, LLaVA inference wrapper
│   ├── evaluation/            # BLEU/ROUGE/CIDEr/BERTScore/RoViST/GROOVIST, Kendall's tau
│   └── pipeline.py           # end-to-end orchestration (encode → order → generate)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_encoding.ipynb           # run on Colab
│   ├── 03_ordering_train.ipynb     # run on Colab (T4)
│   ├── 04_story_generation.ipynb   # run on Colab (A100)
│   └── 05_evaluation.ipynb         # run locally
├── configs/                  # YAML configs for each stage
├── results/                  # generated stories, metric scores, plots
├── proposal_and_thesis/       # your Word docs
└── README.md
```

Push this to a private GitHub repo. Colab notebooks `git pull`/`git push` at the start/end of each session so nothing is Colab-locked. Large binaries (embeddings, checkpoints) go to Google Drive, not GitHub.

---

## 3. Environment Setup (Week 1)

**Local (i5 Windows, no GPU) — for data prep, orchestration code, evaluation, writing:**
```bash
conda create -n sequenceweaver python=3.10
conda activate sequenceweaver
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets pillow pandas numpy scikit-learn matplotlib
pip install bert-score sacrebleu pycocoevalcap  # metrics
```

**Colab — for GPU work:**
- Mount Google Drive first cell of every notebook.
- Pin package versions in a `requirements.txt` and `pip install -r` at the top of each notebook — Colab's preinstalled versions drift and silently break reproducibility.
- Runtime: T4 (free tier) for encoding + ordering training; switch to A100 (Colab Pro, pay-as-you-go) only for LLaVA-13B inference.

**Get access to:**
- VIST dataset (visionandlanguage.net/VIST) — request/download the SIS (Story-in-Sequence) annotations + image URLs.
- LLaVA-1.5-13B weights (Hugging Face `liuhaotian/llava-v1.5-13b`) or use it via a hosted inference API if download/VRAM becomes a blocker — flag this as a fallback now, not later.

---

## 4. Phase-by-Phase Plan

### Phase 0 — Literature review + data prep (Weeks 1–4)
- Read the 5 core papers (VIST, CLIP, SigLIP, LLaVA, VIST-GPT) properly — not just abstracts. Take structured notes per paper: problem, method, dataset, metric, limitation. This becomes your literature review chapter directly.
- Download VIST. Filter to **stories of 2–5 images** (matches your restricted scope) — this itself is a data engineering task: VIST stories are natively 5-image sequences, so for 2–4 image variants you'll need to subsample sub-sequences from the 5-image stories.
- Clean and split: train / val / test (e.g., 80/10/10 at the story level, not image level, to avoid leakage).
- Deliverable: `data/processed/{train,val,test}.json`, each entry = `{story_id, image_paths[], correct_order[], reference_stories[]}`.

**First concrete coding task:** a script that downloads VIST images, filters broken URLs (many VIST image URLs are dead — expect ~10-20% loss), and produces the processed JSON above.

### Phase 1 — Image encoding pipeline (Weeks 3–5, overlaps Phase 0 tail)
- Wrap SigLIP (`google/siglip-so400m-patch14-384` or smaller variant) in a clean function: `encode_image(path) -> np.ndarray[768]`.
- Batch-encode the full processed VIST image set on Colab T4. Cache embeddings to disk (`data/embeddings/{image_id}.npy` or one big `.pt` tensor + index) — you do **not** want to re-run SigLIP every time you touch the ordering model.
- Sanity check: compute cosine similarity between a few known-related image pairs vs. unrelated pairs, confirm it matches expectation (as discussed: ~0.8 for related, ~0.2 for unrelated).
- Deliverable: cached embeddings + a short notebook showing the similarity sanity check (useful figure for your thesis methodology section).

### Phase 2 — Ordering module (Weeks 3–8)
This is your first real novel contribution — give it real time.
- **Model:** lightweight transformer (~5M params) that takes N embeddings (N=2..5, pad/mask for variable length) and outputs a predicted permutation or per-image "position score."
- **Framing options** (pick one, document why in thesis):
  - *Pairwise ranking:* train on "does image A come before image B" for all pairs, then sort by aggregate score.
  - *Listwise (ListNet-style):* predict a score per image, sort by score, compare to ground truth order via a ranking loss.
- **Training data:** shuffle each VIST sequence's images, feed to model, supervise with the known correct order.
- **Metric:** Kendall's τ and/or exact-match accuracy on val set.
- Start with N=5 only (matches native VIST), then generate synthetic 2–4 image sub-sequences (contiguous subsets of the 5) once the base model works — don't build variable-length support before you have a working fixed-length baseline.
- Deliverable: trained checkpoint + `src/ordering/infer.py` that takes a list of embeddings and returns predicted order + a results table (Kendall's τ per N).

### Phase 3 — Story generation pipeline (Weeks 6–11)
- Design the **arc-aware prompt template** — this is your other main contribution. Draft 2–3 candidate templates (e.g., explicit "introduction/rising action/climax/resolution" instructions vs. implicit narrative cues) and keep them as versioned config files, not hardcoded strings — you'll be A/B testing these.
- Build `src/generation/generate_story.py`: takes ordered image paths + prompt template → calls LLaVA-1.5-13B → returns story text.
- Run small-batch tests on Colab A100 first (10–20 sequences) before committing to a full test-set run — LLaVA inference is the most expensive step, don't debug prompts at full scale.
- Deliverable: generated stories for the full test set, saved as JSON (`{story_id, ordered_images, prompt_version, generated_story}`), pulled down to local via Drive.

### Phase 4 — Automatic evaluation (Weeks 8–14, overlaps Phase 3 tail)
Runs entirely locally once stories are saved as text — no GPU needed.
- Implement: BLEU, ROUGE, CIDEr, BERTScore against VIST reference stories.
- Implement or adapt RoViST / GROOVIST (from the VIST-GPT paper) — these are more relevant to "narrative coherence" than BLEU/CIDEr and will make your evaluation section stronger and more defensible.
- Compare across your 2–3 prompt template versions and against a naive per-image-captioning baseline (this baseline is important — it's what makes your "cross-image reasoning" contribution demonstrable rather than asserted).
- Deliverable: results table + plots (`results/automatic_eval.csv`, plots in `results/figures/`).

### Phase 5 — Human evaluation (Weeks 12–16)
- Design a short survey (Google Forms is fine): show a few image sets + generated story + baseline story + human-written VIST story, ask for Likert ratings (coherence, engagement, accuracy) — blind the source labels.
- Recruit n=30–50 (classmates, online panels, department mailing lists).
- Analyze results locally (pandas + scipy for significance testing — e.g., paired t-test or Wilcoxon between your system and baseline).
- Deliverable: survey results + statistical comparison, ready to drop into thesis Results chapter.

### Phase 6 — Writing (Weeks 1–20, continuous)
Don't leave this to the end. Write each chapter as its corresponding phase finishes:
- Week 4: Literature review draft
- Week 8: Methodology (encoding + ordering) draft
- Week 11: Methodology (generation) draft
- Week 14: Results (automatic eval) draft
- Week 16: Results (human eval) draft
- Weeks 17–20: Discussion, conclusion, full revision pass

---

## 5. Your Very First Step (This Week)

Before writing any model code:

1. Create the GitHub repo with the folder structure above.
2. Get VIST access and download a small sample (not the full 210K images yet) — confirm you can actually load images and annotations end-to-end.
3. Write and test the data-filtering script that produces 2–5 image sub-sequences with ground-truth order from VIST's 5-image stories.

Everything downstream (embeddings, ordering, generation) depends on this data format being right, so it's worth getting exactly correct before scaling up.

---

## 6. Key Risks & Fallbacks

| Risk | Mitigation |
|---|---|
| VIST image URLs dead (common — many are 10+ years old) | Budget extra time in Phase 0; consider a fallback subset or PororoSV if loss is too high |
| LLaVA-13B too slow/expensive on Colab A100 | Fall back to LLaVA-7B or a hosted API (e.g., via OpenRouter) for generation; document the substitution |
| Ordering module underperforms | Still a valid thesis finding — report it honestly, pivot emphasis toward prompt engineering as primary contribution |
| Timeline slips | Cut Phase 5 (human eval) sample size down, or narrow automatic eval to BLEU/CIDEr/BERTScore only, dropping RoViST/GROOVIST reimplementation |
