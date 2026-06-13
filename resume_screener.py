"""
AI Resume Screening System
==========================
Matches multiple resumes against a job description using NLP.
Supports: PDF, DOCX, and TXT resume files.

Requirements (install via pip):
    pip install anthropic PyPDF2 python-docx scikit-learn colorama tabulate

Usage:
    python resume_screener.py
"""

import os
import sys
import json
import math
import re
import string
from pathlib import Path
from collections import Counter

# ── third-party imports ──────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    sys.exit("❌  Missing: pip install anthropic")

try:
    from colorama import Fore, Back, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    sys.exit("❌  Missing: pip install colorama")

try:
    from tabulate import tabulate
except ImportError:
    sys.exit("❌  Missing: pip install tabulate")

# optional – graceful fallback if not installed
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESUMES_FOLDER = "resumes"          # put resume files here
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001" # Anthropic model to use

# Common English stop-words for TF-IDF
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","not","no",
    "this","that","these","those","it","its","we","our","you","your","i",
    "my","he","his","she","her","they","their","as","by","from","up","about",
    "into","through","during","including","until","against","among","throughout",
    "also","both","each","few","more","most","other","some","such","than",
    "then","so","just","while","per","via","etc","eg","ie",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    if not PDF_SUPPORT:
        return f"[PDF support unavailable – install PyPDF2: pip install PyPDF2]"
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text_from_docx(path: str) -> str:
    if not DOCX_SUPPORT:
        return "[DOCX support unavailable – install python-docx: pip install python-docx]"
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".docx":
        return extract_text_from_docx(path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        return extract_text_from_txt(path)  # try plain-text fallback


# ─────────────────────────────────────────────────────────────────────────────
# NLP – TF-IDF SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {word: count / total for word, count in counts.items()}


def compute_tfidf_similarity(job_text: str, resume_text: str, all_resume_texts: list[str]) -> float:
    """
    Compute cosine similarity between job description and a resume
    using TF-IDF weighting.
    """
    corpus = [job_text] + all_resume_texts
    corpus_tokens = [tokenize(doc) for doc in corpus]

    # IDF: log((N+1) / (df+1)) + 1  (smoothed)
    N = len(corpus)
    vocab = set(t for doc in corpus_tokens for t in doc)
    idf = {}
    for word in vocab:
        df = sum(1 for doc in corpus_tokens if word in doc)
        idf[word] = math.log((N + 1) / (df + 1)) + 1

    def tfidf_vector(tokens):
        tf = compute_tf(tokens)
        return {word: tf.get(word, 0) * idf[word] for word in vocab}

    job_vec = tfidf_vector(corpus_tokens[0])
    # find which index in all_resume_texts matches resume_text
    idx = all_resume_texts.index(resume_text) + 1
    res_vec = tfidf_vector(corpus_tokens[idx])

    # cosine similarity
    dot = sum(job_vec[w] * res_vec[w] for w in vocab)
    mag_j = math.sqrt(sum(v ** 2 for v in job_vec.values())) or 1
    mag_r = math.sqrt(sum(v ** 2 for v in res_vec.values())) or 1
    return dot / (mag_j * mag_r)


def extract_keywords(text: str, top_n: int = 15) -> list[str]:
    tokens = tokenize(text)
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_n)]


def keyword_overlap(job_keywords: list[str], resume_text: str) -> tuple[list[str], list[str]]:
    resume_tokens = set(tokenize(resume_text))
    matched  = [kw for kw in job_keywords if kw in resume_tokens]
    missing  = [kw for kw in job_keywords if kw not in resume_tokens]
    return matched, missing


# ─────────────────────────────────────────────────────────────────────────────
# AI ANALYSIS  (Anthropic)
# ─────────────────────────────────────────────────────────────────────────────

def ai_analyze_candidate(
    client: anthropic.Anthropic,
    job_desc: str,
    resume_text: str,
    candidate_name: str,
    tfidf_score: float,
) -> dict:
    """
    Ask Claude for a structured analysis of one candidate vs the job.
    Returns a dict with score, strengths, gaps, summary, recommendation.
    """
    prompt = f"""You are an expert recruiter and HR analyst. Analyze this candidate's resume against the job description.

JOB DESCRIPTION:
{job_desc}

CANDIDATE: {candidate_name}
RESUME:
{resume_text}

NLP SIMILARITY SCORE (TF-IDF cosine, 0–1): {tfidf_score:.3f}

Return ONLY a JSON object — no markdown, no explanation — with exactly these fields:
{{
  "ai_score": <integer 0-100>,
  "experience_years": <integer or null>,
  "top_skills_matched": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "critical_gaps": ["gap1", "gap2", "gap3"],
  "strengths": "<one sentence, max 20 words>",
  "concern": "<one sentence, max 20 words>",
  "education_fit": "Strong|Moderate|Weak|Unknown",
  "summary": "<two sentences hiring manager summary>",
  "recommendation": "Strong Yes|Yes|Maybe|No"
}}

Score the candidate 0-100 based on: skills match 40%, experience relevance 30%, education 15%, overall fit 15%."""

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # strip any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: return minimal dict
        return {
            "ai_score": int(tfidf_score * 100),
            "experience_years": None,
            "top_skills_matched": [],
            "critical_gaps": [],
            "strengths": "Could not parse AI response.",
            "concern": "Manual review recommended.",
            "education_fit": "Unknown",
            "summary": raw[:300],
            "recommendation": "Maybe",
        }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RANK_COLORS = [Fore.GREEN, Fore.CYAN, Fore.YELLOW, Fore.WHITE, Fore.WHITE]
REC_COLORS  = {
    "Strong Yes": Fore.GREEN,
    "Yes":        Fore.CYAN,
    "Maybe":      Fore.YELLOW,
    "No":         Fore.RED,
}

def print_banner():
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════╗
║        AI Resume Screening System                    ║
║        Powered by Claude (Anthropic) + NLP           ║
╚══════════════════════════════════════════════════════╝""")
    print(Style.RESET_ALL)


def print_section(title: str):
    print(Fore.CYAN + f"\n{'─'*54}")
    print(Fore.CYAN + Style.BRIGHT + f"  {title}")
    print(Fore.CYAN + f"{'─'*54}" + Style.RESET_ALL)


def bar(score: int, width: int = 20) -> str:
    filled = int(score / 100 * width)
    color  = Fore.GREEN if score >= 70 else (Fore.YELLOW if score >= 50 else Fore.RED)
    return color + "█" * filled + Fore.WHITE + "░" * (width - filled) + Style.RESET_ALL


def print_candidate_result(rank: int, result: dict):
    color   = RANK_COLORS[min(rank - 1, len(RANK_COLORS) - 1)]
    rec_col = REC_COLORS.get(result["ai"]["recommendation"], Fore.WHITE)
    ai      = result["ai"]

    print(color + Style.BRIGHT + f"\n  #{rank}  {result['name']}" + Style.RESET_ALL)
    print(f"      File: {Fore.WHITE}{result['file']}{Style.RESET_ALL}")

    # score bar
    score = ai["ai_score"]
    print(f"      Match: {bar(score)} {color}{score}%{Style.RESET_ALL}")

    # NLP similarity
    sim_pct = int(result["tfidf"] * 100)
    print(f"      NLP similarity (TF-IDF):  {sim_pct}%")

    # recommendation badge
    print(f"      Recommendation: {rec_col}{Style.BRIGHT}{ai['recommendation']}{Style.RESET_ALL}", end="  ")
    print(f"  Education fit: {ai.get('education_fit','?')}", end="  ")
    yrs = ai.get("experience_years")
    if yrs is not None:
        print(f"  Experience: {yrs} yrs")
    else:
        print()

    # summary
    print(f"\n      {Fore.WHITE}{ai['summary']}{Style.RESET_ALL}")

    # skills / gaps
    skills = ai.get("top_skills_matched", [])
    gaps   = ai.get("critical_gaps", [])
    if skills:
        print(f"\n      {Fore.GREEN}✔ Matched:{Style.RESET_ALL} {', '.join(skills)}")
    if gaps:
        print(f"      {Fore.RED}✘ Gaps:   {Style.RESET_ALL} {', '.join(gaps)}")

    print(f"\n      {Fore.GREEN}Strength:{Style.RESET_ALL} {ai.get('strengths','')}")
    print(f"      {Fore.YELLOW}Concern: {Style.RESET_ALL} {ai.get('concern','')}")

    # keyword overlap
    matched_kw  = result.get("matched_keywords", [])
    missing_kw  = result.get("missing_keywords", [])
    if matched_kw:
        print(f"\n      {Fore.CYAN}Keyword hits:{Style.RESET_ALL} {', '.join(matched_kw[:8])}")
    if missing_kw:
        print(f"      {Fore.RED}Keyword miss:{Style.RESET_ALL} {', '.join(missing_kw[:5])}")


def print_summary_table(results: list[dict]):
    print_section("RANKING SUMMARY")
    headers = ["Rank", "Candidate", "AI Score", "NLP Sim", "Recommendation", "Exp (yrs)", "Education"]
    rows = []
    for i, r in enumerate(results, 1):
        ai  = r["ai"]
        rec = ai.get("recommendation", "?")
        col = REC_COLORS.get(rec, Fore.WHITE)
        rows.append([
            f"#{i}",
            r["name"],
            f"{ai['ai_score']}%",
            f"{int(r['tfidf'] * 100)}%",
            col + rec + Style.RESET_ALL,
            str(ai.get("experience_years") or "?"),
            ai.get("education_fit", "?"),
        ])
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def save_report(results: list[dict], job_desc: str, output_path: str = "screening_report.json"):
    report = {
        "job_description_snippet": job_desc[:500],
        "total_candidates": len(results),
        "top_candidate": results[0]["name"] if results else None,
        "candidates": [
            {
                "rank": i + 1,
                "name": r["name"],
                "file": r["file"],
                "ai_score": r["ai"]["ai_score"],
                "tfidf_similarity": round(r["tfidf"], 4),
                "recommendation": r["ai"].get("recommendation"),
                "matched_skills": r["ai"].get("top_skills_matched", []),
                "critical_gaps": r["ai"].get("critical_gaps", []),
                "summary": r["ai"].get("summary", ""),
                "education_fit": r["ai"].get("education_fit"),
                "experience_years": r["ai"].get("experience_years"),
            }
            for i, r in enumerate(results)
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(Fore.GREEN + f"\n  ✔  Report saved to: {output_path}" + Style.RESET_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def get_job_description() -> str:
    """Prompt user to paste or load job description."""
    print(Fore.YELLOW + "\nHow would you like to provide the job description?")
    print("  [1] Paste / type it now")
    print("  [2] Load from a .txt file")
    choice = input(Fore.WHITE + "\nEnter 1 or 2: ").strip()

    if choice == "2":
        path = input("  File path: ").strip().strip('"')
        if not os.path.isfile(path):
            print(Fore.RED + "File not found. Switching to manual entry.")
        else:
            return extract_text(path)

    print(Fore.YELLOW + "\nPaste the job description below.")
    print("When done, type END on a new line and press Enter:\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def collect_resumes() -> list[dict]:
    """
    Auto-load from ./resumes/ folder OR let user specify paths.
    Returns list of {name, file, text}.
    """
    resume_data = []

    # Auto-detect from resumes/ folder
    folder = Path(RESUMES_FOLDER)
    found_files = []
    if folder.exists():
        for ext in ("*.pdf", "*.docx", "*.txt", "*.md"):
            found_files.extend(sorted(folder.glob(ext)))

    if found_files:
        print(Fore.GREEN + f"\n  Found {len(found_files)} file(s) in ./{RESUMES_FOLDER}/:")
        for f in found_files:
            print(f"    • {f.name}")
        use = input(Fore.WHITE + "\nUse these files? [Y/n]: ").strip().lower()
        if use != "n":
            for f in found_files:
                print(f"  Extracting: {f.name} …", end=" ", flush=True)
                text = extract_text(str(f))
                name = f.stem.replace("_", " ").replace("-", " ").title()
                resume_data.append({"name": name, "file": f.name, "text": text})
                print(Fore.GREEN + "done")
            return resume_data

    # Manual entry
    print(Fore.YELLOW + "\nNo resumes folder found. Enter resume file paths manually.")
    print("Press Enter with no input when done.\n")
    while True:
        path = input("  Resume file path (or leave blank to stop): ").strip().strip('"')
        if not path:
            if not resume_data:
                print(Fore.RED + "  Please add at least one resume.")
                continue
            break
        if not os.path.isfile(path):
            print(Fore.RED + f"  File not found: {path}")
            continue
        name = input(f"  Candidate name for '{Path(path).name}': ").strip() or Path(path).stem
        print(f"  Extracting text … ", end="", flush=True)
        text = extract_text(path)
        print(Fore.GREEN + "done")
        resume_data.append({"name": name, "file": Path(path).name, "text": text})

    return resume_data


def main():
    print_banner()

    # ── Job description ───────────────────────────────────────────────────────
    print_section("STEP 1 — JOB DESCRIPTION")
    job_desc = get_job_description()
    if len(job_desc.strip()) < 50:
        sys.exit(Fore.RED + "❌  Job description too short. Please provide more detail.")

    job_keywords = extract_keywords(job_desc, top_n=20)
    print(Fore.CYAN + f"\n  Top keywords extracted: {', '.join(job_keywords[:10])}")

    # ── Resumes ───────────────────────────────────────────────────────────────
    print_section("STEP 2 — RESUMES")
    resumes = collect_resumes()
    if not resumes:
        sys.exit(Fore.RED + "❌  No resumes to screen.")

    # ── NLP: TF-IDF similarity ────────────────────────────────────────────────
    print_section("STEP 3 — NLP ANALYSIS (TF-IDF)")
    all_resume_texts = [r["text"] for r in resumes]
    for r in resumes:
        sim = compute_tfidf_similarity(job_desc, r["text"], all_resume_texts)
        matched_kw, missing_kw = keyword_overlap(job_keywords, r["text"])
        r["tfidf"]            = sim
        r["matched_keywords"] = matched_kw
        r["missing_keywords"] = missing_kw
        print(f"  {r['name']:<30} TF-IDF similarity: {int(sim*100):>3}%  |  Keyword hits: {len(matched_kw)}/{len(job_keywords)}")

    
    # ── Ranking from TF-IDF only ──────────────────────────────────────────────
    print_section("STEP 4 — RANKING (TF-IDF ONLY)")

    for r in resumes:
        r["ai"] = {
            "ai_score": int(r["tfidf"] * 100),
            "recommendation": "Match",
            "summary": f"TF-IDF similarity: {int(r['tfidf']*100)}%",
            "education_fit": "N/A",
            "experience_years": None,
            "top_skills_matched": r["matched_keywords"][:5],
            "critical_gaps": r["missing_keywords"][:5],
            "strengths": "Keyword match found.",
            "concern": "No AI analysis performed."
        }

    # ── Rank ──────────────────────────────────────────────────────────────────
    results = sorted(resumes, key=lambda x: x["ai"]["ai_score"], reverse=True)

    # ── Display ───────────────────────────────────────────────────────────────
    print_section("RESULTS — DETAILED CANDIDATE PROFILES")
    for rank, r in enumerate(results, 1):
        print_candidate_result(rank, r)

    print_summary_table(results)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print_section("SCREENING STATISTICS")
    scores = [r["ai"]["ai_score"] for r in results]
    recommended = [r for r in results if r["ai"].get("recommendation") in ("Strong Yes", "Yes")]
    print(f"  Total screened:     {len(results)}")
    print(f"  Recommended:        {len(recommended)}")
    print(f"  Average AI score:   {int(sum(scores)/len(scores))}%")
    print(f"  Highest score:      {max(scores)}%  ({results[0]['name']})")
    print(f"  Lowest score:       {min(scores)}%")

    # ── Save report ───────────────────────────────────────────────────────────
    save_report(results, job_desc)

    print(Fore.GREEN + Style.BRIGHT + "\n  ✅  Screening complete!\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
