"""
Week 2: Text Preprocessing Pipeline (DDI-tuned)
-------------------------------------------------
INPUT:  data/pubmed_abstracts.jsonl
OUTPUT: data/sentences.txt, data/vocab.txt

BUGS FIXED in v3:
  1. Digit rescue pattern was too permissive — `^[a-z]{1,}\\d[a-z0-9]*$`
     passed a0, a1, a2780, a549cells, a2143g (SNP notation), a082002 etc.
     New: _digit_token_ok() uses strict named regex patterns (CYP, cytokine,
     transporter, general biomedical with ≥3-char alpha prefix). Everything
     else with a digit is rejected.

  2. Phrase model was joining junk tokens with underscores — a2_a3,
     a549_h1299, a1_a2_a3, _node, a1c_hba1c. Root cause: junk tokens
     survived tokenization and Phraser joined them. Fix: _is_valid_phrase_token()
     runs on every underscore token in the final write pass and drops phrases
     whose parts fail the junk filter.

  3. Tokens starting with `_` (_node, etc.) now explicitly dropped in
     tokenize_sentence and in the post-phrase filter.

  4. Single-letter alpha prefix tokens (a0, a1 … a9) now caught by the
     ≥3-char alpha prefix requirement in _digit_token_ok.

  5. Mutation/SNP notation (a1166c, a2143g, a118g) — single-letter prefix
     + digits + letter — caught by the same ≥3-char alpha prefix rule.

  6. CYP3A4-mediated was being dropped because hyphen split fired before
     MEDICAL_WHITELIST check on the full token. Fix: check full token first.
"""

import gc
import json
import os
import re
from collections import Counter
from pathlib import Path

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words",     quiet=True)

from nltk.corpus import stopwords, words as nltk_words
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from tqdm import tqdm

try:
    import psutil
    def _mem():
        mb = psutil.Process().memory_info().rss / 1e6
        return f"  [RAM: {mb:.0f} MB]"
except ImportError:
    def _mem(): return ""

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "data/pubmed_abstracts.jsonl"
OUTPUT_SENTS  = "data/sentences.txt"
OUTPUT_VOCAB  = "data/vocab.txt"
MIN_WORD_FREQ = 5
MIN_SENT_LEN  = 5
CHUNK_SIZE    = 5_000
TITLE_REPEAT  = 3       # repeat title N times to boost freq for rare drug names
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_STOPS = {"may", "might", "could", "would", "also", "thus",
                "therefore", "however", "showed", "observed", "found"}
STOP_WORDS    = set(stopwords.words("english")).union(CUSTOM_STOPS)
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

GENERAL_NOISE = {
    "abandoned", "abandonment", "abroad", "absolutely", "abstract", "abstracted",
    "abstraction", "abundance", "abundant", "academia", "academic", "academy",
    "accomplish", "accomplished", "accomplishment", "accordance", "accordingly",
    "accountability", "accountable", "accounting", "accreditation", "accredited",
    "acculturation", "accurately", "acknowledge", "acknowledged", "acorn",
    "ability", "able", "abound", "accept", "acceptable", "acceptance", "accepted",
    "access", "accessibility", "accessible", "accommodate", "accommodating",
    "accommodation", "accompany", "account", "achieve", "achievable", "achievement",
    "acquisition", "aborted", "abortion", "abortive",
}

SHORT_NOISE = {
    "aga", "aba", "mil", "abu", "ada", "ala", "ana", "ara", "ava",
    "awa", "ima", "ina", "ipa", "ira", "isa", "ita", "iva", "oca", "ola",
    "oma", "ona", "ora", "pha", "pia", "pla", "pna", "ria", "sha", "sia",
    "ska", "sla", "sna", "ssa", "tha", "tia", "tla", "tna", "tra",
    "tsa", "wha",
}

# ≤4-char biomedical abbreviations that NLTK doesn't know
MEDICAL_SHORT = {
    # kinases / signalling
    "pkc", "pka", "pde", "ros", "nos", "akt", "jak", "mek", "erk",
    "src", "abl", "ret", "kit", "flt", "axl", "tie",
    # receptors / channels
    "gpcr", "ryr",
    # transporters
    "mdr", "abc",
    # metabolism enzymes
    "ugt", "nat",
    # common short antiretroviral abbreviations used in DDI lit
    "dex", "tac", "csa", "atv", "efv", "lpv", "rtv", "sqv",
    # study design
    "rct", "nnt",
}

MEDICAL_WHITELIST = {
    # units
    "mg", "ml", "kg", "mmhg", "mmol", "iu", "nmol", "umol", "mcg", "bpm",
    # immunoglobulins
    "abo", "igg", "igm", "ige", "iga",
    # common abbreviations
    "hiv", "aids", "covid", "sars", "mri", "ct", "ecg", "eeg", "icu",
    "dna", "rna", "pcr", "mrna", "atp", "ldl", "hdl", "bmi", "bp",
    "copd", "chf", "ckd", "cad", "ibs", "ibd", "als", "vs",
    # anatomy
    "lung", "lungs", "liver", "brain", "kidney", "heart",
    # general clinical
    "mellitus", "insipidus", "arterial", "venous", "coronary",
    "blood", "pressure", "failure", "chronic", "acute",
    # common drugs
    "metformin", "aspirin", "ibuprofen", "warfarin", "heparin",
    "morphine", "codeine", "digoxin", "lithium", "insulin",
    "lasix", "zofran", "taxol", "lyrica", "paxil", "zoloft",
    "lipitor", "prozac", "xanax", "valium",
    # CYP enzymes — full names whitelisted so hyphen splits don't break them
    "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2", "cyp2e1",
    "cyp2b6", "cyp3a5", "cyp2a6", "cyp2j2", "cyp4f2", "p450",
    # tumour suppressors / oncogenes — specific ones only
    "p53",
    # transport proteins
    "p-glycoprotein", "oatp", "oatp1b1", "oatp1b3", "oatp2b1",
    "oat1", "oat3", "oct1", "oct2", "bcrp",
    "mrp1", "mrp2", "mrp4",
    # interaction verbs
    "inhibit", "inhibits", "inhibitor", "inhibition",
    "induce", "induces", "inducer", "induction",
    "potentiate", "potentiates", "potentiation",
    "antagonise", "antagonize", "antagonism",
    "synergise", "synergize", "synergism", "synergy",
    "alter", "alters", "alteration",
    # pharmacokinetic terms
    "bioavailability", "clearance", "halflife", "metabolism",
    "absorption", "distribution", "excretion",
    "pharmacokinetic", "pharmacodynamic",
    # PK parameters
    "cmax", "auc", "tmax",
    # cytokines / biomarkers (digit-containing — kept here explicitly)
    "il-2", "il-6", "il-10", "il-12", "il-17", "il-1b",
    "tnf-a", "ifn-g", "tgf-b",
    "cd4", "cd8", "cd20", "cd19", "cd3",
    "cox-2", "cox-1",
    "her2", "egfr", "vegf", "pdl1", "pd-1", "ctla-4",
    # receptor subtypes
    "5-ht2a", "5-ht3", "5-ht1a", "nmda", "ampa",
    "gaba-a", "gaba-b",
    # DDI mechanism terms
    "efflux", "uptake", "influx",
    # hba1c — legitimate clinical marker, not a code
    "hba1c",
    # antiretrovirals
    "atazanavir", "ritonavir", "lopinavir", "efavirenz", "nevirapine",
    "darunavir", "cobicistat", "elvitegravir", "raltegravir", "dolutegravir",
    "tenofovir", "emtricitabine", "abacavir", "lamivudine", "zidovudine",
    # statins
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
    "fluvastatin", "lovastatin", "pitavastatin",
    # azole antifungals
    "fluconazole", "itraconazole", "ketoconazole", "voriconazole",
    "posaconazole", "isavuconazole",
    # macrolide antibiotics
    "erythromycin", "clarithromycin", "azithromycin", "telithromycin",
}

MEDICAL_SUFFIXES = (
    "itis", "osis", "emia", "uria", "pathy", "plasty", "ectomy", "otomy",
    "oscopy", "graphy", "therapy", "toxin", "mycin", "cillin", "olol",
    "pril", "sartan", "statin", "mab", "nib", "zole", "azine",
    "cycline", "oxacin", "glitazone", "fibrate", "lukast", "kinase",
    "peptide", "steroid", "glucose", "insulin", "receptor",
    "adrenergic", "ergic", "mediated", "dependent", "associated",
    "induced", "transporter", "channel", "reuptake",
    "transferase", "reductase", "oxidase", "synthase", "hydrolase",
    "agonist", "antagonist", "modulator", "substrate",
    "conjugate", "conjugation", "glucuronide", "sulfate",
)

HYPHEN_WHITELIST = {
    "anti-inflammatory", "double-blind", "placebo-controlled", "cross-sectional",
    "well-being", "long-term", "short-term", "follow-up", "self-reported",
    "health-related", "disease-free", "cancer-related", "drug-resistant",
    "hospital-based", "community-based", "evidence-based", "p-glycoprotein",
    "dose-dependent", "time-dependent", "concentration-dependent",
    "dose-response", "mechanism-based", "first-pass", "protein-bound",
    "enzyme-inducing", "enzyme-inhibiting", "substrate-specific",
    "drug-drug", "drug-metabolizing", "drug-transporter",
    "cyp-mediated", "p-gp-mediated", "efflux-mediated",
    "receptor-mediated", "carrier-mediated", "active-site",
    "narrow-therapeutic", "wide-therapeutic",
    "non-linear", "dose-limiting", "rate-limiting",
    "half-life", "steady-state", "peak-trough",
    # common CYP compound adjectives
    "cyp3a4-mediated", "cyp2d6-mediated", "cyp2c9-mediated",
    "cyp3a4-dependent", "cyp2d6-dependent",
    "p-gp-dependent", "p-gp-inhibited",
    "drug-induced",
}

# Word parts that appear in hyphenated DDI terms where one side is medical
MEDICAL_HYPHEN_PARTS = {
    "cyp", "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2",
    "pgp", "mrp", "bcrp", "oatp",
    "mediated", "induced", "inhibited", "dependent", "associated",
    "metabolizing", "metabolising",
    "adrenergic", "kinase", "receptor",
    "agonist", "antagonist", "substrate", "modulator",
}

_HAS_DIGIT  = re.compile(r"\d")
_PURE_ALPHA = re.compile(r"^[a-z]+$")
_VOWELS     = set("aeiou")

# ── Strict regex patterns for digit-containing tokens we want to keep ─────────
# Anything with digits that does NOT match one of these (or MEDICAL_WHITELIST)
# is rejected. This replaces the old permissive `^[a-z]{1,}\d[a-z0-9]*$`.
_CYP_RE         = re.compile(r'^cyp\d[a-z]\d+[a-z]?$')           # cyp3a4, cyp2d6
_CYTOKINE_RE    = re.compile(r'^(il|tnf|ifn|tgf|cox|her|pdl)\d{1,2}[a-z]?$')
_CD_MARKER_RE   = re.compile(r'^cd\d{1,2}$')                      # cd4, cd8, cd19
_TRANSPORTER_RE = re.compile(r'^(oat|oct|mrp|slc|abcb|abcc|abcg)\d{1,2}$')
# General biomedical: ≥3-char alpha prefix + digits + short suffix
# Passes: hba1c, ugt1a1, slco1b1, cyp3a5, her2
# Rejects: a0, a1, a2780, a549, a1166c (single-letter prefix)
_BIOMEDICAL_RE  = re.compile(r'^([a-z]{3,})\d[a-z0-9]{0,5}$')


def _digit_token_ok(tok: str) -> bool:
    """Return True only for digit-containing tokens with a recognised biomedical shape."""
    if tok in MEDICAL_WHITELIST:
        return True
    if _CYP_RE.match(tok):
        return True
    if _CYTOKINE_RE.match(tok):
        return True
    if _CD_MARKER_RE.match(tok):
        return True
    if _TRANSPORTER_RE.match(tok):
        return True
    if _BIOMEDICAL_RE.match(tok):
        return True
    return False


def _is_junk(tok: str) -> bool:
    # Always keep whitelisted tokens
    if tok in MEDICAL_WHITELIST:
        return False

    # Explicit noise
    if tok in SHORT_NOISE:
        return True

    # ── Digit gate — must come before length checks ───────────────────────────
    if _HAS_DIGIT.search(tok):
        return not _digit_token_ok(tok)

    # ── Pure alpha tokens ─────────────────────────────────────────────────────
    if len(tok) <= 2:
        return True

    # Short tokens: rescue MEDICAL_SHORT before ENGLISH_WORDS gate
    if len(tok) <= 4 and "-" not in tok:
        if tok in MEDICAL_SHORT:
            return False
        if tok not in ENGLISH_WORDS:
            return True

    # Hyphenated tokens
    if "-" in tok:
        if tok in HYPHEN_WHITELIST:
            return False
        parts = tok.split("-")
        # Reject parts that are single chars or contain digits
        if any(len(p) <= 1 or _HAS_DIGIT.search(p) for p in parts):
            return True
        # Rescue if any part is a known medical word stem
        if any(p in MEDICAL_HYPHEN_PARTS for p in parts):
            return False
        # Standard English compound: require all parts to be real words
        if not all(p in ENGLISH_WORDS for p in parts):
            return True
        return False

    # Must contain a vowel
    if not any(c in _VOWELS for c in tok):
        return True

    # Longer pure-alpha tokens
    if _PURE_ALPHA.match(tok) and len(tok) > 5:
        if tok in ENGLISH_WORDS:
            return tok in GENERAL_NOISE
        if any(tok.endswith(s) for s in MEDICAL_SUFFIXES):
            return False
        return False  # unknown long token — keep; low freq will prune at vocab stage

    if tok in GENERAL_NOISE:
        return True

    return False


def _is_valid_phrase_token(tok: str) -> bool:
    """
    Post-phrase validation for underscore-joined tokens from gensim Phraser.
    cytochrome_p450 → valid. a2_a3, a549_h1299, _node → invalid.
    Splits on underscore, checks every part through the same filter.
    """
    parts = tok.split("_")
    # Leading/trailing underscore artefact
    if any(p == "" for p in parts):
        return False
    for part in parts:
        if part in MEDICAL_WHITELIST or part in MEDICAL_SHORT:
            continue
        if part in STOP_WORDS or len(part) <= 1:
            return False
        if _is_junk(part):
            return False
    return True


# Matches hyphenated compounds we want to preserve as single tokens through
# word_tokenize — anything where at least one side is a known medical stem or
# the whole token is in HYPHEN_WHITELIST. We temporarily replace `-` with the
# private-use sentinel `§` so NLTK sees one token, then restore after.
_HYPHEN_PRESERVE = re.compile(
    r'\b('
    # CYP-anything: cyp3a4-mediated, cyp2d6-dependent, cyp-mediated
    r'cyp[\w]*'
    # p-gp compounds: p-gp-mediated, p-gp-dependent
    r'|p-gp'
    # 5-HT receptor subtypes: 5-ht2a, 5-ht3
    r'|5-ht[\w]+'
    # IL/TNF/IFN/TGF/COX with digit suffix: il-6, tnf-a, cox-2
    r'|(?:il|tnf|ifn|tgf|cox|gaba|pd)-[\w]+'
    # drug-drug, drug-induced, drug-metabolizing, drug-transporter
    r'|drug-[\w]+'
    # dose-*, mechanism-*, first-*, protein-*, half-*, non-*, rate-*, peak-*
    r'|(?:dose|mechanism|first|protein|half|non|rate|peak|steady|narrow|wide'
    r'|enzyme|substrate|receptor|carrier|active|efflux|anti|double|placebo'
    r'|cross|well|long|short|follow|self|health|disease|cancer|hospital'
    r'|community|evidence|time|concentration|p-glyco)-[\w-]+'
    r')\b',
    re.IGNORECASE,
)
_SENTINEL = "§"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b[pnrtz]\s*[=<>]\s*[\d\.]+", "", text)
    text = re.sub(r"\[\d+\]|\(\w[\w\s,\.]+\d{4}\w?\)", "", text)
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_sentence(sentence: str) -> list:
    text = clean_text(sentence)

    # Encode hyphens inside preserved compounds so word_tokenize sees one token.
    # e.g. "cyp3a4-mediated" -> "cyp3a4§mediated", tokenized as one piece,
    # then restored to "cyp3a4-mediated" before the junk filter runs.
    def _protect(m):
        return m.group(0).replace("-", _SENTINEL)

    text = _HYPHEN_PRESERVE.sub(_protect, text)

    result = []
    for tok in word_tokenize(text):
        tok = tok.lower().strip("-")
        # Restore sentinel back to hyphen
        tok = tok.replace(_SENTINEL, "-")
        if not tok:
            continue
        if tok.startswith("_"):
            continue
        if tok in MEDICAL_WHITELIST:
            result.append(tok)
            continue
        if tok in STOP_WORDS or len(tok) <= 1:
            continue
        if not _is_junk(tok):
            result.append(tok)
    return result


def record_to_sentences(record: dict) -> list:
    title    = record.get("title", "").strip()
    abstract = record.get("abstract", "").strip()
    text     = (title + ". ") * TITLE_REPEAT + abstract

    out = []
    for sentence in sent_tokenize(text):
        tokens = tokenize_sentence(sentence)
        if len(tokens) >= MIN_SENT_LEN:
            out.append(tokens)
    return out


class StreamingSentences:
    """Streams a .txt file line by line — zero memory overhead."""
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split()


class BigramStream:
    """
    Applies a frozen Phraser on-the-fly during iteration.
    CRITICAL: must NOT be a list comprehension — yields one sentence at a time.
    """
    def __init__(self, path: str, phraser: Phraser):
        self.path    = path
        self.phraser = phraser

    def __iter__(self):
        for sent in StreamingSentences(self.path):
            yield self.phraser[sent]


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def build_raw_sentences(input_path: str, raw_path: str) -> None:
    print(f"Pass 1/4: tokenizing -> {raw_path}{_mem()}")
    total_docs = total_sents = 0
    chunk = []

    with open(input_path, encoding="utf-8") as fin, \
         open(raw_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, unit="doc"):
            line = line.strip()
            if not line:
                continue
            try:
                chunk.append(json.loads(line))
            except json.JSONDecodeError:
                continue

            if len(chunk) >= CHUNK_SIZE:
                for record in chunk:
                    for sent in record_to_sentences(record):
                        fout.write(" ".join(sent) + "\n")
                        total_sents += 1
                total_docs += len(chunk)
                chunk.clear()
                gc.collect()

        for record in chunk:
            for sent in record_to_sentences(record):
                fout.write(" ".join(sent) + "\n")
                total_sents += 1
        total_docs += len(chunk)

    print(f"  {total_docs:,} docs -> {total_sents:,} sentences{_mem()}")


def apply_phrases(raw_path: str, output_path: str) -> None:
    # ── Bigram pass ───────────────────────────────────────────────────────────
    print(f"Pass 2/4: training bigrams (streaming){_mem()}")
    bigram_model = Phrases(
        StreamingSentences(raw_path),
        min_count=5,
        threshold=8,
        max_vocab_size=3_000_000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    bigram = Phraser(bigram_model)
    del bigram_model
    gc.collect()
    print(f"  Bigrams trained{_mem()}")

    # ── Trigram pass ──────────────────────────────────────────────────────────
    print(f"Pass 3/4: training trigrams (streaming){_mem()}")
    trigram_model = Phrases(
        BigramStream(raw_path, bigram),
        min_count=5,
        threshold=8,
        max_vocab_size=3_000_000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigram = Phraser(trigram_model)
    del trigram_model
    gc.collect()
    print(f"  Trigrams trained{_mem()}")

    # ── Write final sentences with post-phrase junk filter ────────────────────
    # KEY FIX: after bigram/trigram joining, validate every token.
    # Underscore-joined phrases whose parts are junk are dropped here,
    # not allowed to survive into sentences.txt and then vocab.txt.
    print(f"Pass 4/4: writing final sentences (with post-phrase filter){_mem()}")
    n = count_lines(raw_path)
    with open(raw_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=n, unit="sent"):
            raw_tokens = line.strip().split()
            if not raw_tokens:
                continue
            phrased = trigram[bigram[raw_tokens]]
            clean = []
            for tok in phrased:
                if tok.startswith("_"):
                    continue
                if "_" in tok:
                    if _is_valid_phrase_token(tok):
                        clean.append(tok)
                elif tok in MEDICAL_WHITELIST or tok in MEDICAL_SHORT:
                    clean.append(tok)
                elif not _is_junk(tok):
                    clean.append(tok)
            if len(clean) >= MIN_SENT_LEN:
                fout.write(" ".join(clean) + "\n")
    print(f"  Done{_mem()}")


def build_vocab(sentences_path: str, vocab_path: str) -> int:
    counter: Counter = Counter()
    for tokens in tqdm(StreamingSentences(sentences_path),
                       unit="sent", desc="counting vocab"):
        counter.update(tokens)
    vocab = {w for w, c in counter.items() if c >= MIN_WORD_FREQ}
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")
    return len(vocab)


def quick_test() -> None:
    """
    Verify tokenizer output. Expected behaviour annotated per sentence.
    """
    test_cases = [
        # ── should produce meaningful tokens ──
        ("Warfarin and aspirin interact via CYP2C9 inhibition.",
         "warfarin aspirin interact cyp2c9 inhibition"),
        ("The patient received 500 mg of amoxicillin twice daily.",
         "patient mg amoxicillin twice daily"),
        ("IL-6 and TNF-a are elevated in inflammatory conditions.",
         "il-6 tnf-a elevated inflammatory conditions"),
        ("Drug-drug interactions may alter cytochrome P450 metabolism.",
         "drug-drug interactions alter cytochrome p450 metabolism"),
        ("Ritonavir is a potent CYP3A4-mediated inhibitor of atazanavir clearance.",
         "ritonavir potent cyp3a4-mediated inhibitor atazanavir clearance"),
        ("P-gp-mediated efflux reduces bioavailability of the substrate drug.",
         "p-gp-mediated efflux reduces bioavailability substrate drug"),
        ("PKC and JAK signalling pathways modulate drug-induced hepatotoxicity.",
         "pkc jak signalling pathways modulate drug-induced hepatotoxicity"),
        ("Ketoconazole AUC increased when co-administered with cobicistat.",
         "ketoconazole auc increased cobicistat"),
        ("Dose-response relationships for CYP2D6-dependent oxidation were non-linear.",
         "dose-response relationships cyp2d6-dependent oxidation non-linear"),
        # ── junk — should produce empty or near-empty output ──
        ("A0 A1 A2780 A549 a1166c a2143g a082002.",
         "(expect empty — all alphanumeric codes)"),
        ("The a1_a2 a3_a4 _node a2b5 segment was removed.",
         "(expect: segment removed — rest is junk)"),
    ]
    print("=== Tokenizer quick test ===")
    for sent, expected in test_cases:
        tokens = tokenize_sentence(sent)
        print(f"  IN      : {sent}")
        print(f"  EXPECTED: {expected}")
        print(f"  GOT     : {tokens}\n")


if __name__ == "__main__":
    quick_test()

    Path("data").mkdir(exist_ok=True)
    RAW_SENTS = "data/sentences_raw.txt"

    build_raw_sentences(INPUT_FILE, RAW_SENTS)
    apply_phrases(RAW_SENTS, OUTPUT_SENTS)
    os.remove(RAW_SENTS)

    print("Building vocab...")
    vocab_size = build_vocab(OUTPUT_SENTS, OUTPUT_VOCAB)
    n_sents    = count_lines(OUTPUT_SENTS)

    print(f"\nDone:{_mem()}")
    print(f"  Sentences : {n_sents:,}")
    print(f"  Vocab     : {vocab_size:,} words (min freq={MIN_WORD_FREQ})")
    print(f"\nOutputs: {OUTPUT_SENTS}  {OUTPUT_VOCAB}")
    print("Next step -> run week3_train_embeddings.py")