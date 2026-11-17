"""Streamlit front end for the ML book RAG assistant.

Run with:  streamlit run app.py
"""

import time

import ollama
import streamlit as st

from rag import RAGPipeline, Settings
from rag.generator import NOT_FOUND
from rag.pipeline import cited_ids, normalise_refusal

st.set_page_config(page_title="ML Book Assistant", page_icon="📘", layout="wide")

EXAMPLES = [
    "What is the difference between Ridge and Lasso regression?",
    "How does k-fold cross-validation work?",
    "Why should scaling be done inside a pipeline?",
    "What are core samples in DBSCAN?",
    "Who won the FIFA World Cup in 2022?",
]


@st.cache_resource(show_spinner="Loading the embedding model and vector index (first run builds it, ~15s)…")
def load_pipeline(llm_model: str) -> RAGPipeline:
    rag = RAGPipeline(Settings(llm_model=llm_model))
    rag.ensure_index()
    return rag


def available_models() -> list[str]:
    try:
        names = [m["model"] for m in ollama.list()["models"]]
    except Exception:
        return []
    return [n for n in names if not n.endswith(":cloud")]


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    models = available_models()
    if not models:
        st.error("Can't reach Ollama. Start it (`ollama serve`) and pull a model: `ollama pull llama3.1:8b`.")
        st.stop()
    default = models.index("llama3.1:8b") if "llama3.1:8b" in models else 0
    llm_model = st.selectbox("LLM (Ollama)", models, index=default)
    k = st.slider("Chunks to retrieve (k)", 1, 10, Settings.top_k)
    st.caption(
        f"Embeddings: `{Settings.embed_model}`  \n"
        f"Chunks: {Settings.chunk_size} tokens, {Settings.chunk_overlap} overlap  \n"
        "Vector DB: ChromaDB (cosine)"
    )
    st.divider()
    st.caption("Try one of these:")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

rag = load_pipeline(llm_model)

# ---------------- main ----------------
st.title("📘 ML Book Assistant")
st.write(
    "Ask a question about *Introduction to Machine Learning with Python*. "
    "Answers come only from the book, with page references. If the book doesn't cover it, the assistant says so."
)

question = st.text_input("Your question", key="question", placeholder="e.g. What is overfitting?")

if question:
    t0 = time.perf_counter()
    hits = rag.retrieve(question, k)
    retrieval_secs = time.perf_counter() - t0

    st.subheader("Answer")
    placeholder = st.empty()
    text = ""
    t1 = time.perf_counter()
    try:
        for token in rag.generator.stream(question, hits):
            text += token
            placeholder.markdown(text + "▌")
    except Exception as e:  # Ollama not running, model missing, etc.
        st.error(f"Generation failed: {e}")
        st.stop()
    text = normalise_refusal(text.strip())
    placeholder.markdown(text)
    gen_secs = time.perf_counter() - t1

    cited = cited_ids(text)
    if text != NOT_FOUND and cited:
        st.markdown("**Sources**")
        for i in cited:
            if 0 < i <= len(hits):
                h = hits[i - 1]
                st.markdown(f"- **[S{i}]** {h.chapter} - {h.page_label}")
    elif text != NOT_FOUND:
        st.warning("The answer has no citations, so it may not be grounded in the book. Treat it with caution.")

    st.caption(f"Retrieval {retrieval_secs:.2f}s · generation {gen_secs:.1f}s · model {llm_model} · k={k}")

    with st.expander(f"Retrieved chunks ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            mark = "✅ cited" if i in cited else ""
            st.markdown(f"**S{i}** · similarity {h.score:.3f} · {h.chapter} · {h.page_label} {mark}")
            st.text(h.text[:1200] + (" …" if len(h.text) > 1200 else ""))
            st.divider()
