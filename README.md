# AI Note/PDF Summarizer (AI Automation)

A small web app that summarizes PDFs, DOCX, and TXT with either:
- **OpenAI API** (if `OPENAI_API_KEY` is set), or
- **Local lightweight model** (`t5-small` via `transformers`) — no API key required.

## Quick Start

### 1) Create & activate a venv
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> If you don't have a GPU, it's fine — this uses CPU (just slower).

### 3) (Optional) Use OpenAI for better quality & speed
```bash
export OPENAI_API_KEY="your_key_here"   # Windows PowerShell: $env:OPENAI_API_KEY="your_key_here"
```

### 4) Run the app
```bash
python app.py
```
Open http://localhost:5000 and upload a .pdf/.txt/.docx.

## Notes
- Large documents are split into chunks and summarized with a **map-reduce** approach.
- If PDFs are scanned images, extract text first (e.g., with OCR tools) or convert to searchable PDFs.
- You can switch the local model in `summarize_with_local()` to `sshleifer/distilbart-cnn-12-6` for better quality (heavier).
