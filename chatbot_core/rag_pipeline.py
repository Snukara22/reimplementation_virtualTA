import os
import re
import random
import logging
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional
from wordsegment import load, segment


import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PATHS & GLOBALS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXTBOOK_PDF = PROJECT_ROOT / "saas-csc517-textbook.pdf"

# Where logs will be stored
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# In-memory logs
interaction_logs: List[Dict[str, Any]] = []
feedback_logs: List[Dict[str, Any]] = []

# Chunking parameters
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# LOGGING UTILS


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON object as one line in a .jsonl text file."""
    try:
        import json
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logging.error("Error writing to JSONL log file: %s", e)



# HELPER FUNCTIONS (HINT, REFLECTION, NEXT STEP)


NEXT_STEP_SUGGESTIONS = [
    "Try writing a 3-line code example using this concept.",
    "Explain this idea aloud as if teaching a friend.",
    "Identify which chapter this idea appears in and skim that page.",
    "Think of how this concept appears in homework or the project.",
]


def clean_sentence(s: str) -> str:
    s = " ".join(s.split())
    s = re.sub(r"([,:;])(?!\s)", r"\1 ", s)
    return s
def prettify_text(txt: str) -> str:
    """
    Heavy-duty text cleaner that reconstructs spacing for merged words
    from PDFs like the SaaS textbook.
    """

    # Normalize weird hyphens and unicode spacing
    txt = txt.replace("‐", "-").replace("–", "-").replace("—", "-")
    txt = txt.replace("“", '"').replace("”", '"')

    # 0) Fix run-together lowercase-uppercase boundaries
    txt = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", txt)

    # 1) Insert space after punctuation when missing
    txt = re.sub(r"([.,;:!?])(?=[A-Za-z])", r"\1 ", txt)

    # 2) Insert space between letters and numbers
    txt = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", txt)
    txt = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", txt)


    try:
        load()
        segmented = segment(txt)
        txt = " ".join(segmented)
    except:
        # if wordsegment not installed, fallback to naive spacing
        pass

    # 4) Remove double spaces
    txt = " ".join(txt.split())

    # 5) Capitalize sentence starts
    sentences = re.split(r'(?<=[.?!])\s+', txt)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    txt = " ".join(sentences)

    return txt




def choose_next_step() -> str:
    return random.choice(NEXT_STEP_SUGGESTIONS)


def build_reflection_html() -> str:
    return (
        "<ol>"
        "<li>What is the question really asking you to understand?</li>"
        "<li>Which CSC 517 topic or concept is this related to?</li>"
        "<li>After reading the answer, what is still unclear?</li>"
        "</ol>"
    )


def build_hint_from_context(question: str, context: str) -> str:
    if not context:
        return "<i>No hint available yet.</i>"

    sentences = []
    for s in context.replace("\n", " ").split("."):
        s = s.strip()
        if 20 < len(s) < 200:
            sentences.append(prettify_text(s))
        if len(sentences) >= 3:
            break

    if not sentences:
        return (
            "<ul>"
            f"<li>Focus on the main term: <b>{question}</b></li>"
            "<li>Look for definitions or examples in the retrieved text.</li>"
            "<li>Try to restate the idea in your own words.</li>"
            "</ul>"
        )

    items = "".join(f"<li>{s}.</li>" for s in sentences)
    return f"<p>Based on retrieved context:</p><ul>{items}</ul>"



# LOAD TEXTBOOK + BUILD FAISS STORE


@lru_cache(maxsize=1)
def load_textbook_text() -> str:
    """Load entire SaaS PDF into one big string."""
    if not TEXTBOOK_PDF.exists():
        raise FileNotFoundError(f"Cannot find textbook at {TEXTBOOK_PDF}")

    pages = []
    with pdfplumber.open(TEXTBOOK_PDF) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """Build and cache FAISS index over textbook chunks."""
    text = load_textbook_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb



# FALLBACK (NO API KEY) - KEYWORD RETRIEVAL


def _simple_keyword_retrieve(question: str, top_k: int = 3):
    text = load_textbook_text()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    words = re.findall(r"\w+", question.lower())
    keywords = [w for w in words if len(w) > 3] or words

    scored = []
    for idx, para in enumerate(paragraphs):
        score = sum(para.lower().count(k) for k in keywords)
        if score > 0:
            scored.append((score, idx, para))

    scored.sort(reverse=True, key=lambda x: x[0])
    scored = scored[:top_k]

    chunks = [p for (_, _, p) in scored]
    sources = [
        f"Excerpt {i} (raw paragraph #{idx}): {clean_sentence(p[:220])}..."
        for i, (_, idx, p) in enumerate(scored)
    ]

    return chunks, sources



# PROMPT BUILDING + LLM CREATION


def _build_prompt(question: str, first_guess: str, onboarding_pref: str, context: str):
    if onboarding_pref == "concise":
        style = "Give a concise 2–4 sentence explanation with one simple example."
    elif onboarding_pref == "detailed":
        style = "Give a detailed explanation with steps, definitions, and examples."
    else:
        style = "Explain clearly in normal language."

    if first_guess.strip():
        fg = (
            "The student first attempted:\n"
            f"\"\"\"\n{first_guess.strip()}\n\"\"\"\n\n"
            "Correct any misunderstandings and build on what is right.\n\n"
        )
    else:
        fg = "The student did not provide a first guess.\n\n"

    return f"""
You are a CSC 517 teaching assistant. 
Answer ONLY using the textbook context below. 
If the answer is not present, say you are not sure.

{style}

--- BEGIN CONTEXT ---
{context}
--- END CONTEXT ---

{fg}

Now answer the student's question:

Question:
\"\"\"{question.strip()}\"\"\" 
"""


def _get_llm() -> Optional[ChatOpenAI]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return ChatOpenAI(temperature=0.1)
    except:
        return None



# MAIN FUNCTION CALLED BY FLASK


def answer_question(question: str, first_guess: str = "", onboarding_pref: str = "concise"):
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved_chunks": [],
            "hint_html": "",
            "reflection_html": "",
            "next_step": "",
        }

  
    # NO API KEY -> FALLBACK MODE
  
    if not os.getenv("OPENAI_API_KEY"):
        chunks, sources = _simple_keyword_retrieve(question, top_k=3)
        context = "\n\n---\n\n".join(chunks)

        hint = build_hint_from_context(question, context)
        reflection = build_reflection_html()
        next_step = choose_next_step()

        result = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": context or "No similar content found in textbook.",
            "first_guess": first_guess,
            "onboarding_pref": onboarding_pref,
            "mode": "keyword_fallback",
        }

        interaction_logs.append(result)
        _append_jsonl(LOG_DIR / "interaction_log.jsonl", result)

        return {
            "answer": result["answer"],
            "sources": sources,
            "retrieved_chunks": chunks,
            "hint_html": hint,
            "reflection_html": reflection,
            "next_step": next_step,
        }


    # FULL RAG MODE
   
    db = get_vectorstore()
    docs = db.similarity_search(question, k=TOP_K)

    retrieved_chunks = []
    sources = []
    for i, d in enumerate(docs):
        txt = d.page_content or ""
        retrieved_chunks.append(txt)

        meta = d.metadata or {}
        page = meta.get("page") or meta.get("page_num") or meta.get("page_number")
        label = f"Doc {i}" + (f" (page {page+1})" if isinstance(page, int) else "")

        pretty = prettify_text(txt[:240])
        sources.append(f"{label}: {prettify_text(txt[:350])}...")

    context = "\n\n---\n\n".join(retrieved_chunks)

    llm = _get_llm()
    if not llm:
        return {"answer": "Cannot initialize LLM.", "sources": [], "retrieved_chunks": []}

    prompt = _build_prompt(question, first_guess, onboarding_pref, context)
    response = llm.invoke(prompt)
    answer_text = getattr(response, "content", str(response))

    # Log interaction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer_text,
        "first_guess": first_guess,
        "onboarding_pref": onboarding_pref,
        "mode": "rag_llm",
    }

    interaction_logs.append(log_entry)
    _append_jsonl(LOG_DIR / "interaction_log.jsonl", log_entry)

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": [prettify_text(c) for c in retrieved_chunks],
        "hint_html": build_hint_from_context(question, context),
        "reflection_html": build_reflection_html(),
        "next_step": choose_next_step(),
    }


# ============================================================
# FEEDBACK LOGGING
# ============================================================

def log_feedback(payload: Dict[str, Any]) -> None:
    """Called from Flask /feedback endpoint."""
    fb = dict(payload)
    fb["timestamp"] = datetime.now().isoformat()

    feedback_logs.append(fb)
    _append_jsonl(LOG_DIR / "feedback_log.jsonl", fb)
