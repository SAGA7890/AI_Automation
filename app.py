import os
import io
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Optional imports — loaded lazily only when needed
def safe_import_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except Exception as e:
        return None

def safe_import_pymupdf():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception:
        return None

def safe_import_docx():
    try:
        import docx
        return docx
    except Exception:
        return None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB upload limit
ALLOWED_EXT = {'.pdf', '.txt', '.docx'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT

# -------- File readers --------
def read_txt(f: io.BytesIO) -> str:
    return f.read().decode('utf-8', errors='ignore')

def read_pdf(f: io.BytesIO) -> str:
    fitz = safe_import_pymupdf()
    if not fitz:
        raise RuntimeError("PyMuPDF not installed. Please install 'pymupdf' to read PDFs.")
    text = []
    with fitz.open(stream=f.read(), filetype='pdf') as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def read_docx(f: io.BytesIO) -> str:
    docx = safe_import_docx()
    if not docx:
        raise RuntimeError("python-docx not installed. Please install 'python-docx' to read .docx files.")
    document = docx.Document(f)
    return "\n".join(p.text for p in document.paragraphs)

def extract_text(file_storage) -> str:
    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()
    file_bytes = io.BytesIO(file_storage.read())
    if ext == '.txt':
        return read_txt(file_bytes)
    elif ext == '.pdf':
        return read_pdf(file_bytes)
    elif ext == '.docx':
        return read_docx(file_bytes)
    else:
        raise ValueError("Unsupported file type")

# -------- Chunking helper --------
def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> list:
    # Simple sentence boundary split with overlap
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            # start new with some overlap from previous
            if overlap and chunks:
                tail = chunks[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks

# -------- Summarizers --------
def summarize_with_openai(chunks, openai_model="gpt-4o-mini"):
    # Uses the OpenAI REST API if OPENAI_API_KEY is set
    import requests, os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None  # signal to fallback
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    summaries = []
    for ch in chunks:
        payload = {
            "model": openai_model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a concise academic note summarizer. Output 4-8 bullet points capturing key ideas, definitions, data, and conclusions. Avoid fluff."
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text:\n\n{ch}"
                }
            ]
        }
        # Using /v1/responses to be SDK-agnostic
        resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Extract text
        content = ""
        try:
            content = data["output"][0]["content"][0]["text"]
        except Exception:
            # fallback for different shapes
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        summaries.append(content.strip())
    # Reduce step
    payload = {
        "model": openai_model,
        "input": [
            {"role": "system", "content": "Combine the points into a single clean summary with sections: Overview, Key Points (bullets), Takeaways (3 bullets)."},
            {"role": "user", "content": "\\n\\n".join(summaries)}
        ]
    }
    resp = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    try:
        final_text = data["output"][0]["content"][0]["text"]
    except Exception:
        final_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return final_text.strip()

def summarize_with_local(chunks, model_name="t5-small"):
    pipeline = safe_import_transformers()
    if not pipeline:
        raise RuntimeError("transformers not installed. Please install 'transformers' and 'torch' for local summarization.")
    summarizer = pipeline("summarization", model=model_name)
    summaries = []
    for ch in chunks:
        # T5 expects task prefix
        text = "summarize: " + ch
        out = summarizer(text, max_length=180, min_length=60, do_sample=False)
        summaries.append(out[0]["summary_text"])
    # Reduce step
    combined = " ".join(summaries)
    combined_out = summarizer("summarize: " + combined, max_length=220, min_length=80, do_sample=False)
    return combined_out[0]["summary_text"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        text = extract_text(f)
        if not text.strip():
            return jsonify({"error": "No text extracted. Is the file scanned or empty?"}), 400
        chunks = chunk_text(text, max_chars=3500, overlap=150)

        # Try OpenAI first; if no key, fallback to local
        try:
            summary = summarize_with_openai(chunks)
        except Exception as e:
            summary = None

        if not summary:
            summary = summarize_with_local(chunks)

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
