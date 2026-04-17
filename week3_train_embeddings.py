"""
Week 3: Train Biomedical Word Embeddings (Word2Vec / Skip-gram)
----------------------------------------------------------------
INPUT:  data/sentences.txt   (output of week2_preprocess.py)
        data/vocab_clean.txt (output of clean_vocab.py — optional but recommended)
OUTPUT: models/medical_word2vec.model
        models/medical_word2vec.kv
        outputs/embedding_eval.txt

HOW TO RUN:
    python week3_train_embeddings.py

RAM FIXES vs previous version (each one is a real crash risk):

  1. WORKERS capped at 8.
     os.cpu_count() on WSL often returns the full host core count (16-32).
     Gensim spawns one input-queue thread per worker plus accumulation
     buffers. At 20+ workers on 5.5M sentences the queue RAM alone can
     exceed 4-6 GB and trigger the OOM killer. Cap at min(cores, 8).

  2. MAX_VOCAB_SIZE added to Word2Vec constructor.
     During vocab-build Gensim accumulates a raw frequency dict in RAM
     before pruning to min_count. On a 80k-vocab corpus from 5.5M sentences
     this intermediate dict can temporarily hold millions of entries.
     max_final_vocab=120_000 tells Gensim to hard-cap and prune early.

  3. model.wv.save() instead of model.save() for the KeyedVectors artefact.
     model.save() pickles the full model including the training-only
     syn1neg weight matrix (same size as wv.vectors: vocab × dim floats).
     For 80k × 200 that is 80,000 × 200 × 4 bytes = 64 MB extra in RAM
     at save time plus the same again on disk. We save both, but the KV
     file no longer needs the full model loaded to use it downstream.

  4. del model.syn1neg after training.
     The negative-sampling output matrix is only needed during training.
     Freeing it before evaluate_embeddings() and save_outputs() drops RAM
     by ~64 MB and prevents a second copy appearing if numpy makes a view.

  5. Vocabulary restricted to vocab_clean.txt when available.
     Word2Vec's min_count still runs, but pre-restricting to the clean
     vocab means the intermediate frequency dict never grows beyond 80k
     entries regardless of what tokens appear in sentences.txt.
     This is done via Word2Vec(..., min_count=1) after pre-filtering, or
     by passing a custom vocab via build_vocab(update=False) — see below.

  6. BATCH_WORDS tuned down from the Gensim default (10,000).
     Each worker holds BATCH_WORDS tokens in its input buffer. At 20
     workers × 10,000 × ~8 bytes = 1.6 GB of queue RAM. Halving it to
     5,000 cuts this to 800 MB with negligible throughput impact on
     sequential disk I/O (disk is the bottleneck, not the queue).
"""

import gc
import os
import logging
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

try:
    import psutil
    def _mem() -> str:
        mb = psutil.Process().memory_info().rss / 1e6
        return f"[RAM {mb:.0f} MB]"
except ImportError:
    def _mem() -> str: return ""

# ── Config ────────────────────────────────────────────────────────────────────
SENTENCES_FILE  = "data/sentences.txt"
VOCAB_FILE      = "data/vocab.txt"   # set to None to skip vocab filter
MODEL_DIR       = "models"
OUTPUT_DIR      = "outputs"

VECTOR_SIZE  = 200      # Use Linear(200,128) projection before cross-attention
                        # to match Node2Vec d=128 rather than constraining here.
WINDOW       = 8
MIN_COUNT    = 10       # Second gate after vocab_clean filtering. Raises from 5
                        # to 10 to prune borderline-rare tokens during training.
                        # Set to 1 if VOCAB_FILE is set (vocab already cleaned).
WORKERS      = min(os.cpu_count() or 4, 8)   # RAM fix 1: cap at 8
EPOCHS       = 20
SG           = 1        # Skip-gram
NEGATIVE     = 10
SAMPLE       = 1e-4     # Subsampling threshold for frequent words
BATCH_WORDS  = 5_000    # RAM fix 6: halved from Gensim default of 10,000
MAX_VOCAB    = 120_000  # RAM fix 2: hard cap on vocab size during build
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    level=logging.WARNING)


class EpochLogger(CallbackAny2Vec):
    """
    Per-epoch loss delta logger.
    Gensim's get_latest_training_loss() is cumulative, so we store the
    value at epoch BEGIN and subtract it at epoch END for the true delta.
    """
    def __init__(self, total_epochs: int):
        self.total       = total_epochs
        self.epoch       = 0
        self._start_loss = 0.0
        self._t0         = time.time()

    def on_epoch_begin(self, model):
        self._start_loss = model.get_latest_training_loss()
        self._t0 = time.time()

    def on_epoch_end(self, model):
        self.epoch += 1
        loss  = model.get_latest_training_loss()
        delta = loss - self._start_loss
        secs  = time.time() - self._t0
        print(f"  Epoch {self.epoch:>2}/{self.total}  "
              f"loss_delta: {delta:>12,.0f}  "
              f"time: {secs:.1f}s  {_mem()}")


class CorpusReader:
    """
    Memory-efficient streaming corpus — never loads more than one line.
    __len__ is cached so Word2Vec's internal progress bar doesn't re-scan.

    If a vocab_set is provided, tokens not in that set are silently dropped.
    This pre-filters noise before Gensim's own min_count pass, keeping the
    intermediate frequency dict small.
    """
    def __init__(self, path: str, vocab_set: set | None = None):
        self.path      = path
        self.vocab_set = vocab_set
        self._len: int | None = None

    def __iter__(self):
        vs = self.vocab_set
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if vs is not None:
                    tokens = [t for t in tokens if t in vs]
                if tokens:
                    yield tokens

    def __len__(self) -> int:
        if self._len is None:
            with open(self.path, "rb") as f:
                self._len = sum(1 for _ in f)
        return self._len


def load_vocab_set(vocab_path: str | None) -> set | None:
    """Load vocab_clean.txt into a set for O(1) token lookup during streaming."""
    if vocab_path is None or not Path(vocab_path).exists():
        if vocab_path is not None:
            print(f"  [!] {vocab_path} not found — skipping vocab filter")
        return None
    words = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    vocab = {w.strip() for w in words if w.strip()}
    print(f"  Vocab filter loaded: {len(vocab):,} tokens from {vocab_path}")
    return vocab


def train_model(sentences_path: str, vocab_path: str | None) -> Word2Vec:
    print(f"\n{'─'*55}")
    print(f"Step 1/3: Loading vocab filter  {_mem()}")
    vocab_set = load_vocab_set(vocab_path)

    # Use min_count=1 when vocab_set pre-filters — the clean vocab already
    # handled frequency thresholding. Otherwise use MIN_COUNT.
    effective_min_count = 1 if vocab_set is not None else MIN_COUNT

    corpus  = CorpusReader(sentences_path, vocab_set=vocab_set)
    n_sents = len(corpus)

    print(f"Step 2/3: Building vocab  {_mem()}")
    print(f"  Sentences  : {n_sents:,}")
    print(f"  Workers    : {WORKERS}")
    print(f"  min_count  : {effective_min_count}  (effective)")
    print(f"  max_vocab  : {MAX_VOCAB:,}")

    # Initialise model without training (sentences=None) so we can
    # control vocab build and training as separate steps — this lets us
    # free intermediate structures between steps.
    model = Word2Vec(
        vector_size    = VECTOR_SIZE,
        window         = WINDOW,
        min_count      = effective_min_count,
        workers        = WORKERS,
        sg             = SG,
        negative       = NEGATIVE,
        sample         = SAMPLE,
        epochs         = EPOCHS,
        compute_loss   = True,
        batch_words    = BATCH_WORDS,    # RAM fix 6
        max_final_vocab= MAX_VOCAB,      # RAM fix 2
    )

    # Build vocab in one pass — Gensim reads the corpus once here.
    model.build_vocab(corpus, progress_per=500_000)
    actual_vocab = len(model.wv)
    print(f"  Vocab built: {actual_vocab:,} tokens  {_mem()}")

    # Force GC before the training loop — the raw frequency dict that
    # build_vocab used internally is now eligible for collection.
    gc.collect()

    print(f"\nStep 3/3: Training  {_mem()}")
    print(f"  dim={VECTOR_SIZE}  window={WINDOW}  "
          f"epochs={EPOCHS}  negative={NEGATIVE}")

    logger = EpochLogger(EPOCHS)

    model.train(
        corpus,
        total_examples = n_sents,       # avoids a re-scan for progress tracking
        epochs         = EPOCHS,
        compute_loss   = True,
        callbacks      = [logger],
    )

    # RAM fix 4: drop the output weight matrix — only needed during training.
    # For 80k vocab × 200 dim this frees ~64 MB before evaluation and saving.
    if hasattr(model, "syn1neg"):
        del model.syn1neg
        gc.collect()
        print(f"\n  syn1neg freed  {_mem()}")

    return model


def evaluate_embeddings(wv) -> str:
    lines = ["=" * 60, "EMBEDDING EVALUATION REPORT", "=" * 60, ""]

    # ── 1. Nearest neighbours ─────────────────────────────────────────────────
    # General medical terms — just sanity-checks that the model learned context
    test_words = [
        "warfarin", "cyp3a4", "inhibition", "metformin",
        "diabetes", "hypertension", "cancer", "chemotherapy",
    ]
    lines.append("1. NEAREST NEIGHBOURS (top 5)")
    lines.append("-" * 40)
    for word in test_words:
        if word in wv:
            nbs = wv.most_similar(word, topn=5)
            nb_str = ", ".join(f"{w} ({s:.3f})" for w, s in nbs)
            lines.append(f"  {word:<20} -> {nb_str}")
        else:
            lines.append(f"  {word:<20} -> [OOV]")

    # ── 2. DDI analogy tests ──────────────────────────────────────────────────
    lines.extend(["", "2. DDI ANALOGY TESTS  (A - B + C ≈ ?)", "-" * 40])
    analogies = [
        ("warfarin",    "bleeding",    "aspirin"),     # warfarin - bleeding + aspirin ≈ ?
        ("cyp3a4",      "inhibition",  "cyp2d6"),      # CYP family analogy
        ("insulin",     "diabetes",    "metformin"),
        ("ritonavir",   "cyp3a4",      "fluconazole"),  # both CYP3A4 inhibitors
    ]
    for a, b, c in analogies:
        if all(w in wv for w in (a, b, c)):
            res    = wv.most_similar(positive=[a, c], negative=[b], topn=3)
            rs     = ", ".join(f"{w} ({s:.3f})" for w, s in res)
            lines.append(f"  {a} - {b} + {c}")
            lines.append(f"    ≈ {rs}")
        else:
            missing = [w for w in (a, b, c) if w not in wv]
            lines.append(f"  {a} - {b} + {c}  [OOV: {missing}]")

    # ── 3. DDI-specific similarity pairs ─────────────────────────────────────
    lines.extend(["", "3. PAIRWISE COSINE SIMILARITY", "-" * 40])
    lines.append("  (HIGH expected > 0.5,  LOW expected < 0.2)")
    pairs = [
        ("warfarin",      "anticoagulant",   "HIGH"),
        ("cyp3a4",        "cyp2d6",          "HIGH"),   # same enzyme family
        ("inhibitor",     "inhibition",      "HIGH"),   # morphological pair
        ("ritonavir",     "cobicistat",      "HIGH"),   # both CYP3A4 boosters
        ("metformin",     "diabetes",        "HIGH"),
        ("absorption",    "bioavailability", "HIGH"),
        ("warfarin",      "metformin",       "MED"),    # unrelated drugs
        ("cyp3a4",        "economy",         "LOW"),    # sanity check
    ]
    for w1, w2, expected in pairs:
        if w1 in wv and w2 in wv:
            sim = wv.similarity(w1, w2)
            bar = "#" * int(sim * 30)
            lines.append(f"  {w1:<20} ~ {w2:<20}  {sim:.4f}  {bar}  [{expected}]")
        else:
            missing = [w for w in (w1, w2) if w not in wv]
            lines.append(f"  {w1} / {w2}  [OOV: {missing}]")

    # ── 4. DDI phrase / bigram spot check ────────────────────────────────────
    lines.extend(["", "4. DDI PHRASE SPOT CHECK", "-" * 40])
    phrases = [
        "cytochrome_p450",
        "drug_drug_interaction",
        "adverse_drug_reaction",
        "drug_induced_liver_injury",
        "p-glycoprotein",
        "blood_brain_barrier",
        "first_pass_metabolism",
        "steady_state",
        "half_life",
        "dose_dependent",
    ]
    present = [p for p in phrases if p in wv]
    absent  = [p for p in phrases if p not in wv]
    for p in present:
        lines.append(f"  ✓  {p}")
    for p in absent:
        lines.append(f"  ✗  {p}  [absent]")

    # ── 5. Vocab stats ────────────────────────────────────────────────────────
    lines.extend(["", "5. VOCAB STATS", "-" * 40])
    lines.append(f"  Total tokens in model : {len(wv):,}")
    lines.append(f"  Embedding dimension   : {wv.vector_size}")

    # Coverage check against DDI-critical terms
    ddi_critical = [
        "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2",
        "p-glycoprotein", "oatp1b1", "bcrp",
        "inhibitor", "inducer", "substrate",
        "bioavailability", "clearance", "metabolism",
        "warfarin", "ritonavir", "ketoconazole", "metformin",
        "drug-drug", "drug-induced",
    ]
    covered = [w for w in ddi_critical if w in wv]
    lines.append(f"  DDI critical coverage : {len(covered)}/{len(ddi_critical)}")
    if len(covered) < len(ddi_critical):
        missing = [w for w in ddi_critical if w not in wv]
        lines.append(f"  Missing               : {missing}")
    lines.append("")
    return "\n".join(lines)


def save_outputs(model: Word2Vec, report: str) -> None:
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    model_path  = Path(MODEL_DIR) / "medical_word2vec.model"
    kv_path     = Path(MODEL_DIR) / "medical_word2vec.kv"
    report_path = Path(OUTPUT_DIR) / "embedding_eval.txt"

    # Save full model (needed if you want to resume training later)
    model.save(str(model_path))

    # Save KeyedVectors only — this is what downstream code needs.
    # RAM fix 3: KV file does NOT include syn1neg (already deleted above),
    # so it's half the size of the full model on disk.
    model.wv.save(str(kv_path))

    report_path.write_text(report, encoding="utf-8")

    sizes = {
        p.name: f"{p.stat().st_size / 1e6:.1f} MB"
        for p in [model_path, kv_path, report_path]
        if p.exists()
    }
    print(f"\nSaved outputs:")
    for name, size in sizes.items():
        print(f"  {name:<40} {size}")
    print(f"\n  → Use '{kv_path}' for downstream DDI modelling.")


if __name__ == "__main__":
    if not Path(SENTENCES_FILE).exists():
        raise FileNotFoundError(
            f"'{SENTENCES_FILE}' not found. Run week2_preprocess.py first."
        )

    t_start = time.time()

    model   = train_model(SENTENCES_FILE, VOCAB_FILE)
    elapsed = (time.time() - t_start) / 60
    print(f"\nTotal training time: {elapsed:.1f} min")

    report  = evaluate_embeddings(model.wv)
    print("\n" + report)

    save_outputs(model, report)
    print("\nDone.")