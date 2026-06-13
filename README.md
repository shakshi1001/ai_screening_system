# AI Resume Screening System
> Powered by Claude (Anthropic) + TF-IDF NLP | Runs in VS Code terminal

---

## Setup (do this once)

### 1. Get your Anthropic API key
Sign up at https://console.anthropic.com and create an API key.

### 2. Install dependencies
Open a terminal in VS Code (`Ctrl+` `) and run:

```bash
pip install anthropic PyPDF2 python-docx colorama tabulate
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Set your API key

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

**Mac / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Or just paste it when the program asks.

---

## Usage

### Quick start (demo mode)
1. The `resumes/` folder already has 3 sample resumes
2. Run:
```bash
python resume_screener.py
```
3. When asked for the job description, choose option **[2]** and enter:
```
job_description.txt
```

### With your own resumes
1. Drop your `.pdf`, `.docx`, or `.txt` resume files into the `resumes/` folder
2. Run `python resume_screener.py`
3. Paste or load your job description

---

## What it does

| Step | Technology | What happens |
|------|-----------|--------------|
| 1 | File parsing | Extracts text from PDF / DOCX / TXT |
| 2 | TF-IDF NLP | Computes cosine similarity between job and resume |
| 3 | Keyword analysis | Finds matched and missing job keywords |
| 4 | Claude AI | Deep analysis — scores, gaps, strengths, recommendation |
| 5 | Ranking | Sorts all candidates highest → lowest |
| 6 | Report | Saves `screening_report.json` |

---

## Output files

| File | Description |
|------|-------------|
| `screening_report.json` | Full structured results for all candidates |

---

## Project structure

```
resume_screening/
├── resume_screener.py       ← main script
├── requirements.txt
├── job_description.txt      ← sample job description
├── README.md
└── resumes/
    ├── priya_sharma.txt     ← sample resume 1
    ├── marcus_liu.txt       ← sample resume 2
    └── ayesha_noor.txt      ← sample resume 3
```

---

## Supported file formats

| Format | Library | Notes |
|--------|---------|-------|
| `.txt` | built-in | always works |
| `.pdf` | PyPDF2 | text-based PDFs only (not scanned) |
| `.docx`| python-docx | standard Word documents |
| `.md`  | built-in | treated as plain text |
