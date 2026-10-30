import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4

st.set_page_config(page_title="PDF Question Answering Chatbot", page_icon="📖", layout="wide")

CSS = """
<style>
.stApp { background: #ffffff; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] { background: #f4f7fd; border-right: 1px solid #e6ebf5; }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.sb-logo { width: 42px; height: 42px; border-radius: 10px; background: #dbe7ff;
  display: flex; align-items: center; justify-content: center; margin-bottom: 14px; }
.sb-title { font-size: 1.6rem; font-weight: 700; color: #111a3a; line-height: 1.25; margin: 0 0 12px 0; }
.sb-desc { color: #4a5470; font-size: 0.95rem; line-height: 1.5; padding-bottom: 20px;
  border-bottom: 1px solid #dfe5f1; margin-bottom: 24px; }
.step { display: flex; gap: 14px; margin-bottom: 26px; }
.step-num { flex: 0 0 26px; height: 26px; border-radius: 50%; color: #fff; font-weight: 600;
  font-size: 0.85rem; display: flex; align-items: center; justify-content: center; }
.step-title { font-weight: 600; color: #111a3a; font-size: 0.95rem; margin-bottom: 4px; }
.step-text { color: #6b7590; font-size: 0.82rem; line-height: 1.5; }
.sb-footer { color: #4a5470; font-size: 0.8rem; margin-top: 40px; display: flex; align-items: center; gap: 8px; }

.hero { display: flex; align-items: center; justify-content: center; gap: 18px; margin-bottom: 28px; }
.hero h1 { font-size: 2.1rem; font-weight: 700; color: #111a3a; margin: 0; padding: 0; }
.hero p { color: #6b7590; margin: 4px 0 0 0; font-size: 1rem; }

.st-key-upload_card, .st-key-answer_card {
  background: #f6f9ff; border: 1px solid #e1e8f5 !important; border-radius: 12px; padding: 18px 20px;
}
.card-head { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 6px; }
.card-title { font-weight: 600; font-size: 1.1rem; color: #111a3a; }
.card-sub { color: #6b7590; font-size: 0.85rem; margin-top: 2px; }

[data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 1px solid #d9e1f0; border-radius: 10px; }

.stButton button, .stFormSubmitButton button { border-radius: 8px; font-weight: 600; }
.stFormSubmitButton button { background: #3b82f6; border-color: #3b82f6; color: #fff; height: 2.9rem; }
.stFormSubmitButton button:hover { background: #2563eb; border-color: #2563eb; color: #fff; }

.success-box { display: flex; gap: 14px; align-items: center; background: #eefaf2; border: 1px solid #c9ecd6;
  border-radius: 10px; padding: 14px 18px; margin: 18px 0 22px 0; }
.success-icon { width: 26px; height: 26px; border-radius: 50%; background: #16a34a; color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; }
.success-title { color: #15803d; font-weight: 600; font-size: 0.95rem; }
.success-text { color: #3f6b50; font-size: 0.85rem; }

.section-title { display: flex; gap: 10px; align-items: center; font-weight: 600; font-size: 1.15rem;
  color: #111a3a; margin: 8px 0 8px 0; }
.st-key-ask_form [data-testid="stForm"] { border: none; padding: 0; }
.st-key-ask_form input { height: 2.9rem; border-radius: 8px; }

.st-key-answer_box { background: #ffffff; border: 1px solid #e1e8f5 !important; border-radius: 8px; }
.placeholder { color: #8a93ab; font-size: 0.9rem; }
.st-key-answer_card [data-testid="stExpander"] details { background: #eef2fd; border: 1px solid #dfe5f6; border-radius: 8px; }
</style>
"""

ROBOT_SVG = """<svg width="72" height="64" viewBox="0 0 72 64" xmlns="http://www.w3.org/2000/svg">
<circle cx="32" cy="34" r="28" fill="#eef2ff"/>
<line x1="32" y1="8" x2="32" y2="17" stroke="#4f46e5" stroke-width="3"/>
<circle cx="32" cy="7" r="3.5" fill="#4f46e5"/>
<rect x="8" y="27" width="6" height="12" rx="3" fill="#4f46e5"/>
<rect x="50" y="27" width="6" height="12" rx="3" fill="#4f46e5"/>
<rect x="13" y="17" width="38" height="31" rx="11" fill="#6366f1"/>
<rect x="18" y="22" width="28" height="20" rx="8" fill="#1e1b4b"/>
<circle cx="25.5" cy="31" r="3" fill="#a5f3fc"/>
<circle cx="38.5" cy="31" r="3" fill="#a5f3fc"/>
<path d="M27 36.5 q5 3.5 10 0" stroke="#a5f3fc" stroke-width="2" fill="none" stroke-linecap="round"/>
<path d="M62 14 l1.5 4 4 1.5 -4 1.5 -1.5 4 -1.5 -4 -4 -1.5 4 -1.5z" fill="#818cf8"/>
<path d="M66 32 l1 2.5 2.5 1 -2.5 1 -1 2.5 -1 -2.5 -2.5 -1 2.5 -1z" fill="#a5b4fc"/>
</svg>"""

BOOK_SVG = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round"><path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z"/>
<path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z"/></svg>"""

DOC_SVG = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="1.8"
stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
<path d="M14 2v6h6"/><path d="M8 13h8M8 17h5"/></svg>"""

CHAT_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1e3a8a" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-6.4A8 8 0 1 1 21 12z"/>
<path d="M8 12h.01M12 12h.01M16 12h.01"/></svg>"""

SPARKLE_SVG = """<svg width="22" height="22" viewBox="0 0 24 24" fill="#6366f1">
<path d="M12 2l2.2 6.3L20.5 10.5l-6.3 2.2L12 19l-2.2-6.3L3.5 10.5l6.3-2.2z"/>
<path d="M19 15l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" opacity=".7"/></svg>"""

STEPS = [
    ("#3b82f6", "Upload a PDF", "Choose a text-based PDF file from your computer."),
    ("#8b5cf6", "Process the PDF", "Click the button to extract and index the content."),
    ("#10b981", "Ask a question", "Type your question related to the PDF content."),
    ("#ec4899", "Get an answer", "View the AI response and check the source text."),
]


def html(markup):
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def extract_pages(file):
    pages = []
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages, chunk_size, overlap):
    chunks = []
    step = max(chunk_size - overlap, 1)
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if piece:
                chunks.append({"page": p["page"], "text": piece})
            if start + chunk_size >= len(words):
                break
    return chunks


def retrieve(query, embedder, chunks, embeddings, k):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]


def build_prompt(question, hits):
    context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for c, _ in hits)
    return (
        "You are a helpful assistant answering questions about the user's PDF document.\n"
        "Use only the context below. If the answer is not in the context, say you don't know.\n"
        "Cite pages like [page N].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def ask_ollama(prompt, url, model):
    resp = requests.post(
        f"{url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


html(CSS)

with st.sidebar:
    html(f"""
    <div class="sb-logo">{BOOK_SVG}</div>
    <div class="sb-title">PDF Question Answering Chatbot</div>
    <div class="sb-desc">Upload your PDF, ask questions, and get answers using AI.</div>
    """)
    html("".join(
        f'<div class="step"><div class="step-num" style="background:{color}">{n}</div>'
        f'<div><div class="step-title">{title}</div><div class="step-text">{text}</div></div></div>'
        for n, (color, title, text) in enumerate(STEPS, start=1)
    ))
    with st.expander("Model settings"):
        ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
        ollama_model = st.text_input("Model", "llama3.2")
    html('<div class="sb-footer"><span style="color:#e11d48">&#10084;</span> Built with Streamlit</div>')

html(f"""
<div class="hero">
  {ROBOT_SVG}
  <div><h1>PDF Question Answering Chatbot</h1>
  <p>Upload your PDF and ask questions. Get accurate answers powered by AI.</p></div>
</div>
""")

with st.container(key="upload_card"):
    html(f"""
    <div class="card-head">{DOC_SVG}
      <div><div class="card-title">Upload PDF</div>
      <div class="card-sub">Choose a PDF file (text-based) from your computer.</div></div>
    </div>
    """)
    file = st.file_uploader(
        "Drag and drop a PDF here, or click **Upload** to browse your computer",
        type="pdf",
    )
    process = st.button(":material/settings: Process PDF", disabled=file is None)

if process and file is not None:
    with st.spinner("Reading and indexing the PDF..."):
        pages = extract_pages(file)
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            st.session_state.pop("chunks", None)
            st.error("No extractable text found. Scanned PDFs need OCR first.")
        else:
            embeddings = load_embedder().encode(
                [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False
            )
            st.session_state.chunks = chunks
            st.session_state.embeddings = np.asarray(embeddings)
            st.session_state.pop("answer", None)
            st.session_state.pop("hits", None)

ready = "chunks" in st.session_state

if ready:
    html("""
    <div class="success-box"><div class="success-icon">&#10003;</div>
      <div><div class="success-title">PDF processed successfully!</div>
      <div class="success-text">You can now ask questions about your document.</div></div>
    </div>
    """)

html(f'<div class="section-title">{CHAT_SVG} Ask a Question</div>')

with st.container(key="ask_form"):
    with st.form("ask", clear_on_submit=False, border=False):
        col_q, col_btn = st.columns([6, 1], vertical_alignment="bottom")
        question = col_q.text_input(
            "Question", placeholder="Type your question here...", label_visibility="collapsed"
        )
        ask = col_btn.form_submit_button(":material/send: Ask", use_container_width=True)

if ask:
    if not ready:
        st.warning("Please upload and process a PDF first.")
    elif not question.strip():
        st.warning("Please type a question.")
    else:
        hits = retrieve(question, load_embedder(), st.session_state.chunks, st.session_state.embeddings, TOP_K)
        with st.spinner("Thinking..."):
            try:
                answer = ask_ollama(build_prompt(question, hits), ollama_url, ollama_model)
            except requests.RequestException as e:
                answer = (
                    f"Could not reach the LLM at `{ollama_url}` ({e}).\n\n"
                    "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                    f"(`ollama pull {ollama_model}`). The most relevant source text is shown below."
                )
        st.session_state.answer = answer
        st.session_state.hits = hits

with st.container(key="answer_card"):
    html(f'<div class="section-title" style="margin-top:0">{SPARKLE_SVG} Answer</div>')
    with st.container(key="answer_box", border=True):
        if "answer" in st.session_state:
            st.markdown(st.session_state.answer)
        else:
            html('<div class="placeholder">Your answer will appear here...</div>')
    hits = st.session_state.get("hits")
    with st.expander("View Source Text", icon=":material/description:", expanded=bool(hits)):
        st.caption("See the exact text from the PDF used for this answer.")
        if hits:
            for c, score in hits:
                st.markdown(f"**Page {c['page']}** · relevance {score:.2f}")
                st.write(c["text"])
        else:
            st.write("No source text yet.")
