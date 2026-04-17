"""
clean_vocab.py v2 — Whitelist-based vocab cleaning.
-----------------------------------------------------
INPUT:  data/vocab.txt
OUTPUT: data/vocab_clean.txt

FUNDAMENTAL CHANGE FROM v1:
  v1 used a blacklist — only tokens matching specific junk patterns were
  removed. This failed because the tokenizer's `_is_junk` ends with
  `return False` (keep) for any unknown long alpha token, so proper nouns
  (aachen, aalborg_university), HTML entity artifacts (aacute, eacute),
  genomics junk (aaggg_repeat), and foreign words (aabilir, aadhaar) all
  survived into vocab.txt.

  v2 uses a WHITELIST — a token is kept ONLY IF it passes at least one of:
    1. HARD_KEEP  — explicit medical/DDI terms (drugs, enzymes, targets)
    2. MEDICAL_SUFFIX — ends with a known drug/disease/mechanism suffix
    3. BIOMEDICAL_PATTERN — matches a strict regex (CYP, CD marker, etc.)
    4. ENGLISH_WORDS — in NLTK English corpus AND length ≥ 5 AND not in
                        EXTRA_NOISE
    5. HYPHEN_OK — hyphenated compound where all parts ≥ 3 chars AND
                   at least one part is medical or all parts are English
    6. PHRASE_OK — underscore phrase where ALL components pass 1–5

  Unknown = rejected. This is the correct default for biomedical NLP where
  you want a clean, DDI-focused vocabulary rather than an exhaustive corpus
  of everything in PubMed.

Run:
  python3 clean_vocab.py

Prints a per-category breakdown so you can audit what was dropped.
"""

import re
from collections import defaultdict
from pathlib import Path

import nltk
nltk.download("words", quiet=True)
from nltk.corpus import words as nltk_words

INPUT_VOCAB  = "data/vocab.txt"
OUTPUT_VOCAB = "data/vocab_clean.txt"

# ── Build ENGLISH_WORDS once at import time ───────────────────────────────────
ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

# ── Hard keep — always survive regardless of other filters ───────────────────
HARD_KEEP = {
    # PK parameters
    "auc", "cmax", "tmax", "hba1c", "vd", "cl", "fu", "ppb",
    # transport proteins (full and numbered)
    "p-glycoprotein", "oatp", "oatp1b1", "oatp1b3", "oatp2b1",
    "oat", "oat1", "oat3", "oct", "oct1", "oct2", "bcrp",
    "mrp", "mrp1", "mrp2", "mrp4", "mdr1",
    # CYP enzymes
    "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2", "cyp2e1",
    "cyp2b6", "cyp3a5", "cyp2a6", "cyp2j2", "cyp4f2", "p450",
    "cyp3a4-mediated", "cyp2d6-mediated", "cyp2c9-mediated",
    "cyp3a4-dependent", "cyp2d6-dependent",
    # cytokines / biomarkers
    "il-2", "il-6", "il-10", "il-12", "il-17", "il-1b",
    "tnf-a", "ifn-g", "tgf-b",
    "cd4", "cd8", "cd20", "cd19", "cd3",
    "cox-2", "cox-1", "her2", "egfr", "vegf", "pdl1", "pd-1", "ctla-4",
    # receptor subtypes
    "5-ht2a", "5-ht3", "5-ht1a", "nmda", "ampa", "gaba-a", "gaba-b",
    # units
    "mg", "ml", "kg", "mmhg", "mmol", "iu", "nmol", "umol", "mcg", "bpm",
    # immunoglobulins
    "igg", "igm", "ige", "iga",
    # abbreviations
    "hiv", "aids", "covid", "sars", "dna", "rna", "pcr", "mrna", "atp",
    "ldl", "hdl", "bmi", "bp", "copd", "chf", "ckd", "cad", "ibs", "ibd",
    # DDI mechanism
    "efflux", "uptake", "influx", "bioavailability",
    "pharmacokinetic", "pharmacodynamic",
    "inhibitor", "inhibition", "inducer", "induction",
    "potentiation", "antagonism", "synergy", "synergism",
    "metabolism", "clearance", "absorption", "distribution", "excretion",
    # important processes not always in NLTK
    "glucuronidation", "hydroxylation", "demethylation", "methylation",
    "sulfation", "glucuronide", "hydroxylate", "dealkylation",
    "autoinduction", "autoinhibition",
    "enterohepatic", "presystemic", "transporter-mediated",
    "pharmacogenomics", "pharmacogenetic",
    "drug-drug", "drug-induced", "drug-metabolizing",
    "p-gp-mediated", "efflux-mediated", "receptor-mediated",
    "dose-response", "half-life", "steady-state",
    # kinases / signalling (short forms not in ENGLISH_WORDS)
    "pkc", "pka", "jak", "mek", "erk", "akt", "ros", "nos", "pde",
    "src", "abl", "gpcr",
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
    # other common drugs
    "metformin", "aspirin", "ibuprofen", "warfarin", "heparin",
    "morphine", "codeine", "digoxin", "lithium", "insulin",
    # common clinical terms not always in NLTK
    "mellitus", "pharmacokinetics", "pharmacodynamics",
    "hepatotoxicity", "nephrotoxicity", "cardiotoxicity", "neurotoxicity",
    "cytotoxicity", "genotoxicity", "immunotoxicity",
    "contraindication", "coadministration", "comedication",
    "polypharmacy", "multimorbidity", "comorbidity",
}

# ── Medical drug/disease suffixes — token ending in these is kept ─────────────
MEDICAL_SUFFIXES = (
    # disease
    "itis", "osis", "emia", "uria", "pathy", "algia",
    # procedure
    "plasty", "ectomy", "otomy", "oscopy", "graphy", "stomy",
    # drug classes
    "toxin", "mycin", "cillin", "olol", "pril", "sartan", "statin",
    "mab", "nib", "zole", "azine", "cycline", "oxacin",
    "glitazone", "fibrate", "lukast", "dipine", "tidine", "navir",
    "vudine", "tegravir", "ciclovir", "conazole",
    # biological
    "kinase", "peptide", "steroid", "receptor", "agonist", "antagonist",
    "modulator", "substrate", "transporter", "transferase",
    "reductase", "oxidase", "synthase", "hydrolase", "lipase",
    # pharmacological processes
    "adrenergic", "ergic", "mediated", "dependent", "associated",
    "induced", "channel", "reuptake",
    "conjugate", "conjugation", "glucuronide", "sulfate",
    # therapy
    "therapy", "therapies",
)

# ── Strict biomedical regex patterns for digit-containing tokens ──────────────
_CYP_RE         = re.compile(r'^cyp\d[a-z]\d+[a-z]?$')
_CYTOKINE_RE    = re.compile(r'^(il|tnf|ifn|tgf|cox|her|pdl)\d{1,2}[a-z]?$')
_CD_MARKER_RE   = re.compile(r'^cd\d{1,2}$')
_TRANSPORTER_RE = re.compile(r'^(oat|oct|mrp|slc|abcb|abcc|abcg|ugt)\d{1,2}[a-z]?$')
_BIOMEDICAL_RE  = re.compile(r'^([a-z]{3,})\d[a-z0-9]{0,5}$')

# Old _BIOMEDICAL_RE used `len(prefix) >= 3` which passed aaa1, aac2, aav2.
# This allowlist restricts to known biomedical abbreviation families only.
KNOWN_BIOMEDICAL_PREFIXES = frozenset({
    # CYP family
    "cyp",
    # Phase-II enzymes: glucuronosyltransferases, sulfotransferases, acetyltransferases
    "ugt", "sult", "nat",
    # Glutathione S-transferases
    "gst", "gsta", "gstm", "gstp", "gstt",
    # SLC / ABC transporters
    "slco", "slc", "abcb", "abcc", "abcg",
    # Haemoglobin subtypes
    "hba", "hbb",
    # Monoamine oxidases
    "maoa", "maob",
    # Alcohol / aldehyde dehydrogenases
    "adh", "aldh",
    # Pharmacogenomics enzymes
    "tpmt", "dpyd", "nudt", "por",
    # Kinases with digit suffix (akt1, akt2, jak1, jak2...)
    "akt", "jak", "mek", "erk", "pik", "map",
    # Growth-factor receptors
    "fgfr", "pdgfr", "vegfr",
})

_HAS_DIGIT  = re.compile(r"\d")

# Word parts that make a hyphenated compound medical (one match = ok)
MEDICAL_HYPHEN_PARTS = {
    "cyp", "cyp3a4", "cyp2d6", "cyp2c9", "cyp2c19", "cyp1a2",
    "pgp", "mrp", "bcrp", "oatp", "p-gp",
    "mediated", "induced", "inhibited", "dependent", "associated",
    "metabolizing", "metabolising", "based",
    "adrenergic", "kinase", "receptor",
    "agonist", "antagonist", "substrate", "modulator",
    "inflammatory", "resistant", "related", "free",
    "blind", "controlled", "sectional", "reported",
    "term", "linear", "limiting", "life",
    "pass", "bound", "state", "trough",
}

# General academic/non-DDI words to exclude even if in ENGLISH_WORDS
EXTRA_NOISE = {
    "additionally", "aforementioned", "albeit", "alongside", "amongst",
    "anyway", "appreciably", "approximately", "arguably", "ascertain",
    "attained", "attributed", "broadly", "categorized", "characterized",
    "collectively", "comparatively", "comprising", "concurrently",
    "conducted", "conferred", "considerably", "consistently", "constituted",
    "conversely", "correspondingly", "currently", "decreased", "demonstrated",
    "depicted", "described", "detected", "determined", "disclosed",
    "documented", "dramatically", "elaborated", "emerged", "employed",
    "encompassed", "encountered", "endeavored", "enhanced", "enrolled",
    "ensured", "established", "evaluated", "evidenced", "examined",
    "exhibited", "explored", "expressed", "facilitated", "furthermore",
    "generated", "highlighted", "illustrated", "implemented", "implied",
    "importantly", "included", "incorporated", "indicated", "investigated",
    "involved", "likewise", "maintained", "meanwhile", "mentioned",
    "mitigated", "moreover", "notably", "obtained", "occurred", "outlined",
    "performed", "presented", "previously", "primarily", "principally",
    "proceeded", "provided", "published", "quantified", "recently",
    "recorded", "regarding", "reported", "represented", "respectively",
    "resulted", "revealed", "reviewed", "significantly", "similarly",
    "specifically", "subsequently", "summarized", "supported", "thereby",
    "thereafter", "therein", "throughout", "typically", "ultimately",
    "underscored", "undertaken", "utilized", "validated", "verified",
    "whereas", "wherein", "whereby", "worldwide",
    # non-DDI anatomy unlikely to appear in interaction sentences
    "adipose", "alveolar", "amygdala", "aortic", "appendix",
    "cerebellar", "cerebrospinal", "cervical", "cochlear",
    "conjunctival", "corneal", "dermal", "duodenal", "epididymal",
    "epithelial", "follicular", "ganglionic", "glomerular",
    "ileal", "inguinal", "interstitial", "jejunal",
    "lacrimal", "laryngeal", "lymphatic", "mandibular", "mesenteric",
    "nasal", "oropharyngeal", "ovarian",
    "pericardial", "peritoneal", "pleural", "prostatic",
    "retinal", "salivary", "scrotal", "seminal", "sinusoidal",
    "splenic", "sternal", "synovial", "testicular", "thoracic",
    "thymic", "tonsillar", "tracheal", "uterine", "vaginal", "vesical",
    # generic study-design terms
    "covariate", "demography", "enrolment",
    "exclusion", "inclusion", "intention", "longitudinal",
    "multicenter", "multivariate", "observational",
    "questionnaire", "stratified", "subgroup", "univariate",
}


# ── Core classifier ───────────────────────────────────────────────────────────

def _digit_ok(tok: str) -> bool:
    """True only for digit-containing tokens with a recognised biomedical shape."""
    if tok in HARD_KEEP:
        return True
    if _CYP_RE.match(tok):
        return True
    if _CYTOKINE_RE.match(tok):
        return True
    if _CD_MARKER_RE.match(tok):
        return True
    if _TRANSPORTER_RE.match(tok):
        return True
    # Only allow general biomedical shape if prefix is in the explicit allowlist.
    # This blocks aaa1, aac2, aav2, aal116 (prefix not in allowlist).
    m = _BIOMEDICAL_RE.match(tok)
    if m and m.group(1) in KNOWN_BIOMEDICAL_PREFIXES:
        return True
    return False


def _plain_ok(tok: str) -> bool:
    """
    Whitelist gate for a single token with no underscores.
    Returns True if the token should be kept.
    """
    # Hard keep
    if tok in HARD_KEEP:
        return True

    # Digit-containing: strict pattern required
    if _HAS_DIGIT.search(tok):
        return _digit_ok(tok)

    # Must be at least 4 chars for pure-alpha tokens
    if len(tok) < 4:
        return False

    # Medical suffix — keep regardless of ENGLISH_WORDS
    if any(tok.endswith(s) for s in MEDICAL_SUFFIXES):
        return True

    # Must be in ENGLISH_WORDS (minimum 5 chars to avoid "aa", "aah", "aal")
    if len(tok) >= 5 and tok in ENGLISH_WORDS:
        return tok not in EXTRA_NOISE

    # 4-char tokens only if in ENGLISH_WORDS
    if len(tok) == 4 and tok in ENGLISH_WORDS:
        return tok not in EXTRA_NOISE

    # Unknown — REJECT. This is the key change from v1.
    # The tokenizer kept unknown long alpha tokens by default; we reverse that.
    return False


def _hyphen_ok(tok: str) -> str | None:
    """
    Returns keep-reason string or None (reject) for a hyphenated token.
    """
    if tok in HARD_KEEP:
        return "hard_keep"

    parts = tok.split("-")

    # Any part that's a single char or empty → reject
    if any(len(p) < 2 for p in parts):
        return None

    # Digit-containing parts: use _digit_ok
    for p in parts:
        if _HAS_DIGIT.search(p) and not _digit_ok(p) and p not in HARD_KEEP:
            return None

    # If any part is a known medical stem, we just need all other parts to be
    # valid English words or medical stems
    has_medical_part = any(p in MEDICAL_HYPHEN_PARTS or
                           any(p.endswith(s) for s in MEDICAL_SUFFIXES)
                           for p in parts)

    all_english = all(len(p) >= 3 and (p in ENGLISH_WORDS or p in HARD_KEEP or
                                        p in MEDICAL_HYPHEN_PARTS or
                                        any(p.endswith(s) for s in MEDICAL_SUFFIXES))
                      for p in parts)

    if has_medical_part and all_english:
        return "hyphen_medical"
    if all_english:
        return "hyphen_english"

    return None


def classify(tok: str):
    """Returns (keep: bool, reason: str)."""

    # ── Underscore phrases ────────────────────────────────────────────────────
    if "_" in tok:
        # Check the full phrase against HARD_KEEP first — this rescues things
        # like 5-ht2a_receptor whose hyphenated component confuses part-by-part
        # checking (5-ht2a splits to ["5","ht2a"] where "5" has len 1).
        if tok in HARD_KEEP:
            return True, "hard_keep"
        parts = tok.split("_")
        if any(p == "" for p in parts):
            return False, "empty_phrase_part"
        # Each part must independently pass — either as plain or hyphenated
        for p in parts:
            if "-" in p:
                # Component is itself hyphenated — check it, but also allow
                # it if the plain token (hyphen stripped) is in HARD_KEEP
                if p in HARD_KEEP:
                    continue
                if _hyphen_ok(p) is None:
                    return False, "junk_phrase_part"
            elif not _plain_ok(p):
                return False, "junk_phrase_part"
        return True, "valid_phrase"

    # ── Hyphenated compounds ──────────────────────────────────────────────────
    if "-" in tok:
        reason = _hyphen_ok(tok)
        if reason:
            return True, reason
        return False, "junk_hyphen"

    # ── Plain tokens ──────────────────────────────────────────────────────────
    if _plain_ok(tok):
        return True, "plain_ok"

    # Categorise the rejection reason for the report
    if _HAS_DIGIT.search(tok):
        return False, "bad_digit_pattern"
    if len(tok) < 4:
        return False, "too_short"
    if tok in EXTRA_NOISE:
        return False, "extra_noise"
    if tok not in ENGLISH_WORDS:
        return False, "not_english_not_medical"  # ← biggest bucket
    return False, "english_noise"


def main():
    input_path  = Path(INPUT_VOCAB)
    output_path = Path(OUTPUT_VOCAB)

    if not input_path.exists():
        print(f"[!] {INPUT_VOCAB} not found. Run week2_preprocess.py first.")
        return

    tokens = input_path.read_text(encoding="utf-8").splitlines()
    tokens = [t.strip() for t in tokens if t.strip()]
    print(f"Loaded {len(tokens):,} tokens from {INPUT_VOCAB}")
    print("Classifying...")

    kept    = []
    removed = defaultdict(list)

    for tok in tokens:
        keep, reason = classify(tok)
        if keep:
            kept.append(tok)
        else:
            removed[reason].append(tok)

    # ── Report ────────────────────────────────────────────────────────────────
    total_removed = sum(len(v) for v in removed.values())
    print(f"\n{'─'*65}")
    print(f"{'Category':<38} {'Removed':>7}   {'Examples (first 4)'}")
    print(f"{'─'*65}")
    for label, items in sorted(removed.items(), key=lambda x: -len(x[1])):
        examples = ", ".join(items[:4])
        print(f"  {label:<36} {len(items):>6}   {examples}")
    print(f"{'─'*65}")
    print(f"  {'TOTAL REMOVED':<36} {total_removed:>6}")
    print(f"  {'KEPT':<36} {len(kept):>6}")
    print(f"{'─'*65}")

    # ── Write ─────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n".join(sorted(kept)) + "\n", encoding="utf-8")

    pct = 100 * (len(tokens) - len(kept)) / len(tokens)
    print(f"\nClean vocab written to {OUTPUT_VOCAB}")
    print(f"Reduction: {len(tokens):,} → {len(kept):,} ({pct:.1f}% removed)")

    # ── Spot checks ───────────────────────────────────────────────────────────
    checks = {
        "should be KEPT": [
            "warfarin", "cyp3a4", "inhibition", "atazanavir", "hba1c",
            "hepatotoxicity", "pharmacokinetic", "drug-induced",
            "cytochrome_p450", "amoxicillin", "bioavailability",
            "5-ht2a_receptor", "5-ht3_receptor_antagonists", "ugt1a1",
        ],
        "should be REMOVED": [
            "aachen", "aabilir", "aadhaar", "aaggg_repeat",
            "aacute", "aalborg_university", "a2780", "a549",
            "aaa1", "aac2", "aav2", "aal116", "aaml1831",
        ],
    }
    kept_set = set(kept)
    print()
    for label, examples in checks.items():
        results = []
        for t in examples:
            mark = "✓" if (t in kept_set) == (label == "should be KEPT") else "✗"
            results.append(f"{mark} {t}")
        print(f"{label}:")
        print("  " + "   ".join(results))

    print(f"\nSample 'a' tokens kept (first 30):")
    a_kept = [t for t in kept if t.startswith("a")][:30]
    print("  " + "  ".join(a_kept))


if __name__ == "__main__":
    main()