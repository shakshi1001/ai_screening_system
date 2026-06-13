# AI Resume Screening System
> NLP-Based Resume Screening using TF-IDF Similarity and Keyword Matching

---

## Project Overview

This project automates the initial resume screening process by comparing multiple resumes against a given job description. The system extracts text from resumes, applies Natural Language Processing (NLP) techniques, calculates TF-IDF similarity scores, performs keyword matching, and ranks candidates based on their relevance to the role.

The project demonstrates real-world applications of AI and NLP in recruitment and hiring workflows.

---

## Features

- Upload and analyze multiple resumes
- Supports PDF, DOCX, TXT, and MD files
- Extracts text automatically from resumes
- Uses NLP preprocessing and TF-IDF similarity scoring
- Identifies matching and missing job-related keywords
- Ranks candidates based on relevance to the job description
- Generates a structured screening report

---

## Technologies Used

- Python
- NLP (Text Processing)
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine Similarity
- PyPDF2
- python-docx
- Colorama
- Tabulate

---

## Installation

```bash
pip install PyPDF2 python-docx colorama tabulate
```

Or:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Application

```bash
python resume_screener.py
```

### Steps

1. Provide a job description (paste or load from a file).
2. Upload or place resumes in the `resumes/` folder.
3. The system extracts text from all resumes.
4. TF-IDF similarity and keyword matching are performed.
5. Candidates are ranked based on relevance scores.
6. Results are displayed and saved as a report.

---

## Workflow

```text
Job Description
       ↓
Resume Text Extraction
       ↓
Text Preprocessing
       ↓
TF-IDF Similarity Analysis
       ↓
Keyword Matching
       ↓
Candidate Ranking
       ↓
Screening Report
```

---

## Project Structure

```text
resume_screening/
├── resume_screener.py
├── requirements.txt
├── README.md
└── resumes/
```

---

## Learning Outcomes

- Natural Language Processing (NLP)
- Text Similarity Analysis
- Resume Screening Automation
- AI Applications in Recruitment
- Information Retrieval Techniques
