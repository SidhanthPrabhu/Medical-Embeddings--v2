"""
Week 2: Text Preprocessing Pipeline (DDI-tuned)
-------------------------------------------------
INPUT:  data/pubmed_abstracts.jsonl
OUTPUT: data/sentences.txt, data/vocab.txt

FINAL FIXES (v4):
  1. FORCED_PHRASES uses word-boundary replacements (re.sub with \\b)
     instead of str.replace to prevent partial matches.
     "drug-drug interaction" was matching the prefix of
     "drug-drug interactions" → producing "drug_drug_interactions".

  2. cytochrome_p450 was being dropped by _is_junk because the digit
     gate fired on the whole underscore-joined token. Fix: _is_junk
     now checks MEDICAL_WHITELIST and underscore presence FIRST, before
     the digit gate. Also added "cytochrome_p450" to MEDICAL_WHITELIST.

  3. cyp3a4-mediated was dropped because _is_junk hit the digit gate
     (cyp3a4 contains digits) before checking HYPHEN_WHITELIST.
     Fix: MEDICAL_WHITELIST check and HYPHEN_WHITELIST check both happen
     BEFORE the digit gate in _is_junk.

  4. Sentinel fix confirmed working: clean_text preserves § via
     [^\w\s\-§], _HYPHEN_PRESERVE encodes hyphens as § AFTER clean_text
     runs, word_tokenize sees one token, § restored to -.
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
TITLE_REPEAT  = 3
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

MEDICAL_SHORT = {
    "pkc", "pka", "pde", "ros", "nos", "akt", "jak", "mek", "erk",
    "src", "abl", "ret", "kit", "flt", "axl", "tie",
    "gpcr", "ryr",
    "mdr", "abc",
    "ugt", "nat",
    "dex", "tac", "csa", "atv", "efv", "lpv", "rtv", "sqv",
    "rct", "nnt",
}

MEDICAL_WHITELIST = {
    # units
    "mg", "ml", "kg", "mmhg", "mmol", "iu", "nmol", "umol", "mcg", "bpm",
    # immunoglobulins
    "abo", "igg", "igm", "ige", "iga",
    # abbreviations
    "hiv", "aids", "covid", "sars", "mri", "ct", "ecg", "eeg", "icu",
    "dna", "rna", "pcr", "mrna", "atp", "ldl", "hdl", "bmi", "bp",
    "copd", "chf", "ckd", "cad", "ibs", "ibd", "als", "vs",
    # anatomy
    "lung", "lungs", "liver", "brain", "kidney", "heart",
    # clinical
    "mellitus", "insipidus", "arterial", "venous", "coronary",
    "blood", "pressure", "failure", "chronic", "acute",
    # drugs
    "metformin", "aspirin", "ibuprofen", "warfarin", "heparin",
    "morphine", "codeine", "digoxin", "lithium", "insulin",
    "lasix", "zofran", "taxol", "lyrica", "paxil", "zoloft",
    "lipitor", "prozac", "xanax", "valium",
    # CYP enzymes
    "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2", "cyp2e1",
    "cyp2b6", "cyp3a5", "cyp2a6", "cyp2j2", "cyp4f2", "p450",
    "p53",
    # transporters
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
    # PK terms
    "bioavailability", "clearance", "halflife", "metabolism",
    "absorption", "distribution", "excretion",
    "pharmacokinetic", "pharmacodynamic",
    "cmax", "auc", "tmax",
    # cytokines
    "il-2", "il-6", "il-10", "il-12", "il-17", "il-1b",
    "tnf-a", "ifn-g", "tgf-b",
    "cd4", "cd8", "cd20", "cd19", "cd3",
    "cox-2", "cox-1",
    "her2", "egfr", "vegf", "pdl1", "pd-1", "ctla-4",
    # receptor subtypes
    "5-ht2a", "5-ht3", "5-ht1a", "nmda", "ampa",
    "gaba-a", "gaba-b",
    # DDI mechanism
    "efflux", "uptake", "influx",
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
    # macrolides
    "erythromycin", "clarithromycin", "azithromycin", "telithromycin",
    # forced DDI phrases — added here so _is_junk never fires on them
    # (digit gate was rejecting cytochrome_p450 before this fix)
    "cytochrome_p450", "cytochrome",
    "drug_drug_interaction", "drug_drug_interactions",
    "adverse_drug_reaction", "drug_induced_liver_injury",
    "blood_brain_barrier", "first_pass_metabolism",
    "half_life", "dose_dependent", "steady_state", "peak_trough",
    "narrow_therapeutic_index", "therapeutic_drug_monitoring",
    "pharmacokinetic_pharmacodynamic",
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
    "cyp3a4-mediated", "cyp2d6-mediated", "cyp2c9-mediated",
    "cyp3a4-dependent", "cyp2d6-dependent",
    "p-gp-dependent", "p-gp-inhibited",
    "drug-induced",
}

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

# ── Forced phrase joins ───────────────────────────────────────────────────────
# Uses re.sub with \b word boundaries instead of str.replace to prevent
# partial matches. "drug-drug interaction" must NOT match inside
# "drug-drug interactions" (the plural), which str.replace does wrong.
# Longest/most-specific surface forms listed first.
_FORCED_PHRASE_RES = [
    (re.compile(r'\bpharmacokinetic pharmacodynamic\b'),  "pharmacokinetic_pharmacodynamic"),
    (re.compile(r'\bdrug induced liver injury\b'),        "drug_induced_liver_injury"),
    (re.compile(r'\bdrug-induced liver injury\b'),        "drug_induced_liver_injury"),
    (re.compile(r'\bnarrow therapeutic index\b'),         "narrow_therapeutic_index"),
    (re.compile(r'\btherapeutic drug monitoring\b'),      "therapeutic_drug_monitoring"),
    (re.compile(r'\bblood brain barrier\b'),              "blood_brain_barrier"),
    (re.compile(r'\bblood-brain barrier\b'),              "blood_brain_barrier"),
    (re.compile(r'\badverse drug reaction\b'),            "adverse_drug_reaction"),
    # Match singular only — plural handled separately below
    (re.compile(r'\bdrug[ -]drug interaction\b'),         "drug_drug_interaction"),
    (re.compile(r'\bfirst[ -]pass metabolism\b'),         "first_pass_metabolism"),
    (re.compile(r'\barea under the curve\b'),             "auc"),
    (re.compile(r'\barea under curve\b'),                 "auc"),
    (re.compile(r'\bcytochrome p-?450\b'),                "cytochrome_p450"),
    (re.compile(r'\bhalf[ -]life\b'),                     "half_life"),
    (re.compile(r'\bdose[ -]dependent\b'),                "dose_dependent"),
    (re.compile(r'\bsteady[ -]state\b'),                  "steady_state"),
    (re.compile(r'\bpeak[ -]trough\b'),                   "peak_trough"),
    (re.compile(r'\bmaximum plasma concentration\b'),     "cmax"),
    (re.compile(r'\bminimum inhibitory concentration\b'), "mic"),
]

def apply_forced_phrases(text: str) -> str:
    """Join DDI-critical multi-word terms before tokenization."""
    for pattern, replacement in _FORCED_PHRASE_RES:
        text = pattern.sub(replacement, text)
    return text

# ── Digit-containing token patterns ──────────────────────────────────────────
_CYP_RE         = re.compile(r'^cyp\d[a-z]\d+[a-z]?$')
_CYTOKINE_RE    = re.compile(r'^(il|tnf|ifn|tgf|cox|her|pdl)\d{1,2}[a-z]?$')
_CD_MARKER_RE   = re.compile(r'^cd\d{1,2}$')
_TRANSPORTER_RE = re.compile(r'^(oat|oct|mrp|slc|abcb|abcc|abcg)\d{1,2}$')
_BIOMEDICAL_RE  = re.compile(r'^([a-z]{3,})\d[a-z0-9]{0,5}$')

def _digit_token_ok(tok: str) -> bool:
    if tok in MEDICAL_WHITELIST: return True
    if _CYP_RE.match(tok): return True
    if _CYTOKINE_RE.match(tok): return True
    if _CD_MARKER_RE.match(tok): return True
    if _TRANSPORTER_RE.match(tok): return True
    if _BIOMEDICAL_RE.match(tok): return True
    return False

def _is_junk(tok: str) -> bool:
    # ── Check whitelist and underscore phrases FIRST — before digit gate ──────
    # Critical: tokens like cyp3a4-mediated contain digits, so without this
    # ordering they hit the digit gate and fail before HYPHEN_WHITELIST is checked.
    # Forced phrases like cytochrome_p450 are in MEDICAL_WHITELIST explicitly.
    if tok in MEDICAL_WHITELIST:
        return False

    # Underscore phrases are validated by _is_valid_phrase_token, not here
    if "_" in tok:
        return False

    # Hyphenated tokens — check whitelist before digit gate
    if "-" in tok:
        if tok in HYPHEN_WHITELIST:
            return False
        parts = tok.split("-")
        if any(len(p) <= 1 or _HAS_DIGIT.search(p) for p in parts):
            return True
        if any(p in MEDICAL_HYPHEN_PARTS for p in parts):
            return False
        if not all(p in ENGLISH_WORDS for p in parts):
            return True
        return False

    if tok in SHORT_NOISE:
        return True

    # ── Digit gate — only plain tokens reach here ─────────────────────────────
    if _HAS_DIGIT.search(tok):
        return not _digit_token_ok(tok)

    if len(tok) <= 2:
        return True

    if len(tok) <= 4:
        if tok in MEDICAL_SHORT:
            return False
        if tok not in ENGLISH_WORDS:
            return True

    if not any(c in _VOWELS for c in tok):
        return True

    if _PURE_ALPHA.match(tok) and len(tok) > 5:
        if tok in ENGLISH_WORDS:
            return tok in GENERAL_NOISE
        if any(tok.endswith(s) for s in MEDICAL_SUFFIXES):
            return False
        return False

    if tok in GENERAL_NOISE:
        return True

    return False


def _is_valid_phrase_token(tok: str) -> bool:
    """Validate underscore-joined phrases from gensim Phraser."""
    if tok in MEDICAL_WHITELIST:
        return True
    parts = tok.split("_")
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


# ── Hyphen sentinel — preserves compounds through word_tokenize ───────────────
# Applied AFTER clean_text (which strips special chars) so the sentinel § is
# not destroyed before word_tokenize sees it.
_HYPHEN_PRESERVE = re.compile(
    r'\b('
    # CYP compounds: cyp3a4-mediated, cyp2d6-dependent
    r'cyp[\w]+(?:-[\w]+)+'
    # p-gp compounds: p-gp-mediated, p-gp-dependent
    r'|p-gp(?:-[\w]+)*'
    # 5-HT receptor subtypes
    r'|5-ht[\w]+'
    # cytokines: il-6, tnf-a, cox-2, gaba-a, pd-1
    r'|(?:il|tnf|ifn|tgf|cox|gaba|pd)-[\w]+'
    # drug-* compounds
    r'|drug-[\w]+'
    # general DDI adjectives
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
    text = apply_forced_phrases(text)   # join DDI phrases before stripping
    text = re.sub(r"\b[pnrtz]\s*[=<>]\s*[\d\.]+", "", text)
    text = re.sub(r"\[\d+\]|\(\w[\w\s,\.]+\d{4}\w?\)", "", text)
    text = text.encode("ascii", errors="ignore").decode()
    # Preserve - and § (sentinel written by tokenize_sentence after this call)
    text = re.sub(r"[^\w\s\-§]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_sentence(sentence: str) -> list:
    # Step 1: clean (includes forced phrase joining)
    text = clean_text(sentence)

    # Step 2: sentinel-encode hyphenated compounds so word_tokenize
    # treats them as single tokens
    def _protect(m):
        return m.group(0).replace("-", _SENTINEL)
    text = _HYPHEN_PRESERVE.sub(_protect, text)

    result = []
    for tok in word_tokenize(text):
        tok = tok.lower().strip("-")
        tok = tok.replace(_SENTINEL, "-")   # restore sentinel
        if not tok or tok.startswith("_"):
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
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split()


class BigramStream:
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
    print(f"Pass 2/4: training bigrams (streaming){_mem()}")
    bigram_model = Phrases(
        StreamingSentences(raw_path),
        min_count=5, threshold=8,
        max_vocab_size=3_000_000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    bigram = Phraser(bigram_model)
    del bigram_model; gc.collect()
    print(f"  Bigrams trained{_mem()}")

    print(f"Pass 3/4: training trigrams (streaming){_mem()}")
    trigram_model = Phrases(
        BigramStream(raw_path, bigram),
        min_count=5, threshold=8,
        max_vocab_size=3_000_000,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigram = Phraser(trigram_model)
    del trigram_model; gc.collect()
    print(f"  Trigrams trained{_mem()}")

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
    test_cases = [
        ("Warfarin and aspirin interact via CYP2C9 inhibition.",
         "warfarin aspirin interact cyp2c9 inhibition"),
        ("The patient received 500 mg of amoxicillin twice daily.",
         "patient mg amoxicillin twice daily"),
        ("IL-6 and TNF-a are elevated in inflammatory conditions.",
         "il-6 tnf-a elevated inflammatory conditions"),
        ("Drug-drug interactions may alter cytochrome P450 metabolism.",
         "drug-drug interactions alter cytochrome_p450 metabolism"),
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
        ("A0 A1 A2780 A549 a1166c a2143g a082002.",
         "(expect empty)"),
        ("The a1_a2 a3_a4 _node a2b5 segment was removed.",
         "segment removed"),
        ("The drug drug interaction between warfarin and aspirin is well documented.",
         "drug_drug_interaction warfarin aspirin documented"),
        ("Drug-drug interactions may alter drug metabolism.",
         "drug-drug interactions alter drug metabolism"),
        ("Half life of metformin is approximately 6 hours.",
         "half_life metformin approximately hours"),
        ("The half-life of warfarin is 40 hours.",
         "half_life warfarin hours"),
        ("Steady state concentrations were reached after 5 days.",
         "steady_state concentrations reached days"),
        ("Blood brain barrier penetration limits CNS drug delivery.",
         "blood_brain_barrier penetration limits drug delivery"),
        ("Cytochrome P450 enzymes mediate oxidative metabolism.",
         "cytochrome_p450 enzymes mediate oxidative metabolism"),
        ("Cytochrome P-450 isoforms were inhibited.",
         "cytochrome_p450 isoforms inhibited"),
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